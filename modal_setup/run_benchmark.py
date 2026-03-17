"""End-to-end benchmark: calibrate routers + measure TIDE speedup vs vanilla HF."""

import modal

from modal_setup.image import build_tide_image
from modal_setup.volumes import VOLUME_MOUNTS

app = modal.App("TIDE-benchmark")
tide_image = build_tide_image(include_bench_deps=False)


@app.function(
    image=tide_image,
    gpu="A10G",
    volumes=VOLUME_MOUNTS,
    timeout=3600,
    memory=32768,
)
def benchmark_model(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    batch_sizes: list[int] = [1, 4, 8, 16],
    seq_len: int = 256,
    thresholds: list[float] = [0.5, 0.7, 0.85, 0.95],
    calibration_samples: int = 200,
    warmup_iters: int = 3,
    bench_iters: int = 10,
):
    """Full pipeline: calibrate -> benchmark throughput -> report."""
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from TIDE.config import TIDEConfig
    from TIDE.calibrate import calibrate
    from TIDE.runtime import TIDERuntime

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Model: {model_name}")
    print()

    # --- Step 1: Load model ---
    print("=" * 60)
    print("STEP 1: Loading model...")
    print("=" * 60)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/root/models",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/root/models")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"Layers: {len(model.model.layers)}")
    print()

    # --- Step 2: Calibrate routers ---
    print("=" * 60)
    print("STEP 2: Calibrating routers...")
    print("=" * 60)
    config = TIDEConfig(
        checkpoint_interval=4,
        calibration_samples=calibration_samples,
        convergence_threshold=0.70,
        router_bottleneck_dim=128,
        min_layers=4,
    )
    t0 = time.time()
    router_path = "/tmp/benchmark_router.pt"
    calibrate(model, tokenizer, config=config, save_path=router_path, device=device)
    print(f"Calibration done in {time.time() - t0:.1f}s")
    print()

    # --- Step 3: Baseline HF throughput ---
    print("=" * 60)
    print("STEP 3: Baseline HF throughput...")
    print("=" * 60)
    hf_results = {}
    for bs in batch_sizes:
        input_ids = torch.randint(0, tokenizer.vocab_size, (bs, seq_len), device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iters):
                model(input_ids)
        torch.cuda.synchronize()

        # Bench
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(bench_iters):
                model(input_ids)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        tps = (bs * seq_len * bench_iters) / elapsed
        hf_results[bs] = tps
        print(f"  BS={bs}: {tps:,.0f} tokens/sec ({elapsed / bench_iters * 1000:.1f}ms/iter)")
    print()

    # --- Step 4: TIDE throughput at various thresholds ---
    print("=" * 60)
    print("STEP 4: TIDE throughput...")
    print("=" * 60)
    all_tide_results = {}

    for threshold in thresholds:
        tide_config = TIDEConfig(
            checkpoint_interval=4,
            exit_threshold=threshold,
            min_layers=4,
        )
        runtime = TIDERuntime(model, router_path, config=tide_config, use_cuda_kernels=False)

        tide_results = {}
        for bs in batch_sizes:
            input_ids = torch.randint(0, tokenizer.vocab_size, (bs, seq_len), device=device)

            # Warmup
            for _ in range(warmup_iters):
                runtime(input_ids)
            torch.cuda.synchronize()

            # Bench
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(bench_iters):
                runtime(input_ids)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            tps = (bs * seq_len * bench_iters) / elapsed
            exit_rate = runtime.last_stats.exit_rate if runtime.last_stats else 0
            speedup = tps / hf_results[bs] if hf_results[bs] > 0 else 0

            tide_results[bs] = {
                "tps": tps,
                "speedup": speedup,
                "exit_rate": exit_rate,
                "ms_per_iter": elapsed / bench_iters * 1000,
            }

        all_tide_results[threshold] = tide_results
        print(f"\n  Threshold={threshold}:")
        for bs, r in tide_results.items():
            print(f"    BS={bs}: {r['tps']:,.0f} tok/s "
                  f"({r['speedup']:.2f}x vs HF) "
                  f"exit_rate={r['exit_rate']:.1%} "
                  f"({r['ms_per_iter']:.1f}ms/iter)")

    # --- Step 5: Exit distribution analysis ---
    print()
    print("=" * 60)
    print("STEP 5: Exit distribution (threshold=0.85)...")
    print("=" * 60)
    tide_config = TIDEConfig(checkpoint_interval=4, exit_threshold=0.85, min_layers=4)
    runtime = TIDERuntime(model, router_path, config=tide_config, use_cuda_kernels=False)
    input_ids = torch.randint(0, tokenizer.vocab_size, (8, seq_len), device=device)
    runtime(input_ids)
    if runtime.last_stats:
        print(runtime.last_stats.summary())
    print()

    # --- Step 6: Quality spot-check ---
    print("=" * 60)
    print("STEP 6: Quality spot-check...")
    print("=" * 60)
    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Explain quantum computing:",
    ]
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Vanilla
        with torch.no_grad():
            vanilla_logits = model(**inputs).logits
        vanilla_pred = tokenizer.decode(vanilla_logits[0, -1].argmax().item())

        # TIDE
        tide_logits = runtime(inputs.input_ids)
        tide_pred = tokenizer.decode(tide_logits[0, -1].argmax().item())

        match = "MATCH" if vanilla_pred.strip() == tide_pred.strip() else "DIFF"
        print(f"  '{prompt}' -> HF:'{vanilla_pred.strip()}' TIDE:'{tide_pred.strip()}' [{match}]")

    # --- Step 7: Decode-time layer skipping ---
    print("=" * 60)
    print("STEP 7: Decode-time layer skipping (generation)...")
    print("=" * 60)
    gen_prompts = [
        "The meaning of life is",
        "Write a function to sort a list:",
        "What is 2 + 2?",
    ]
    for thr in [0.3, 0.5, 0.7]:
        tide_config = TIDEConfig(checkpoint_interval=4, exit_threshold=thr, min_layers=4)
        runtime = TIDERuntime(model, router_path, config=tide_config, use_cuda_kernels=False)
        print(f"\n  Threshold={thr}:")
        for prompt in gen_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Vanilla
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                hf_out = model.generate(
                    inputs.input_ids, max_new_tokens=32,
                    do_sample=False, temperature=None, top_p=None,
                )
            torch.cuda.synchronize()
            hf_time = time.perf_counter() - t0

            # TIDE
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tide_out = runtime.generate(inputs.input_ids, max_new_tokens=32, temperature=0)
            torch.cuda.synchronize()
            tide_time = time.perf_counter() - t0

            hf_text = tokenizer.decode(hf_out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            tide_text = tokenizer.decode(tide_out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            speedup = hf_time / tide_time if tide_time > 0 else 0
            stats = runtime.last_stats
            exit_rate = stats.exit_rate if stats else 0
            exits_info = ""
            if stats and stats.exits_per_layer:
                exits_info = f" exits={dict(sorted(stats.exits_per_layer.items()))}"
            print(f"    '{prompt[:40]}' -> {speedup:.2f}x | exit={exit_rate:.1%}{exits_info}")
            print(f"      HF:   {hf_text[:60]}")
            print(f"      TIDE: {tide_text[:60]}")

    print()

    # --- Summary report ---
    print()
    print("=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    print(f"{'Threshold':<12} {'BS':<6} {'HF tok/s':<12} {'TIDE tok/s':<12} {'Speedup':<10} {'Exit Rate':<10}")
    print("-" * 62)
    for threshold in thresholds:
        for bs in batch_sizes:
            r = all_tide_results[threshold][bs]
            print(f"{threshold:<12.2f} {bs:<6} {hf_results[bs]:<12,.0f} {r['tps']:<12,.0f} "
                  f"{r['speedup']:<10.2f} {r['exit_rate']:<10.1%}")
    print()

    # Memory usage
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


@app.local_entrypoint()
def main(
    model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    large: bool = False,
):
    if large:
        benchmark_model.remote(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            batch_sizes=[1, 4, 8],
            seq_len=256,
            calibration_samples=500,
        )
    else:
        benchmark_model.remote(model_name=model)
