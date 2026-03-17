"""Benchmark TIDE on meta-llama/Llama-3.1-8B-Instruct (A100-80GB).

Calibrates routers with convergence_threshold=0.5 (deeper 32-layer model
converges more readily), then measures TIDE speedup vs vanilla HF at
multiple exit thresholds.

Checkpoints at layers: 3, 7, 11, 15, 19, 23, 27, 31  (interval=4, 32 layers)
"""

import modal

from modal_setup.image import build_tide_image
from modal_setup.volumes import VOLUME_MOUNTS

app = modal.App("TIDE-benchmark-8b")
tide_image = build_tide_image(include_bench_deps=False)


@app.function(
    image=tide_image,
    gpu="A100-80GB",
    volumes=VOLUME_MOUNTS,
    timeout=7200,
    memory=65536,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def benchmark_8b(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    batch_sizes: list[int] = [1, 4, 8],
    seq_len: int = 256,
    thresholds: list[float] = [0.3, 0.5, 0.7, 0.85],
    calibration_samples: int = 300,
    convergence_threshold: float = 0.5,
    checkpoint_interval: int = 4,
    warmup_iters: int = 3,
    bench_iters: int = 10,
):
    """Full pipeline: load gated model -> calibrate -> benchmark -> report."""
    import os
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from TIDE.calibrate import calibrate
    from TIDE.config import TIDEConfig
    from TIDE.runtime import TIDERuntime

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Model: {model_name}")
    print(f"HF token present: {bool(hf_token)}")
    print()

    # ------------------------------------------------------------------ #
    # Step 1: Load model (gated – requires HF token)
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("STEP 1: Loading model...")
    print("=" * 70)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir="/root/models",
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir="/root/models",
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_layers = len(model.model.layers)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"Layers: {n_layers}")
    checkpoint_layers = list(range(checkpoint_interval - 1, n_layers, checkpoint_interval))
    print(f"Checkpoint layers: {checkpoint_layers}")
    print(f"GPU memory after load: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print()

    # ------------------------------------------------------------------ #
    # Step 2: Calibrate routers
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("STEP 2: Calibrating routers...")
    print("=" * 70)
    config = TIDEConfig(
        checkpoint_interval=checkpoint_interval,
        calibration_samples=calibration_samples,
        convergence_threshold=convergence_threshold,
        router_bottleneck_dim=128,
        min_layers=4,
    )
    print(f"  convergence_threshold = {convergence_threshold}")
    print(f"  calibration_samples   = {calibration_samples}")
    print(f"  checkpoint_interval   = {checkpoint_interval}")
    t0 = time.time()
    router_path = "/tmp/benchmark_8b_router.pt"
    calibrate(model, tokenizer, config=config, save_path=router_path, device=device)
    print(f"Calibration done in {time.time() - t0:.1f}s")
    print()

    # ------------------------------------------------------------------ #
    # Step 3: Baseline HF throughput
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("STEP 3: Baseline HF throughput...")
    print("=" * 70)
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

    # ------------------------------------------------------------------ #
    # Step 4: TIDE throughput at various thresholds
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("STEP 4: TIDE throughput...")
    print("=" * 70)
    all_tide_results = {}

    for threshold in thresholds:
        tide_config = TIDEConfig(
            checkpoint_interval=checkpoint_interval,
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
            print(
                f"    BS={bs}: {r['tps']:,.0f} tok/s "
                f"({r['speedup']:.2f}x vs HF) "
                f"exit_rate={r['exit_rate']:.1%} "
                f"({r['ms_per_iter']:.1f}ms/iter)"
            )

    # ------------------------------------------------------------------ #
    # Step 5: Exit distribution per layer
    # ------------------------------------------------------------------ #
    print()
    print("=" * 70)
    print("STEP 5: Exit rates per layer (threshold=0.85)...")
    print("=" * 70)
    tide_config = TIDEConfig(
        checkpoint_interval=checkpoint_interval,
        exit_threshold=0.85,
        min_layers=4,
    )
    runtime = TIDERuntime(model, router_path, config=tide_config, use_cuda_kernels=False)
    input_ids = torch.randint(0, tokenizer.vocab_size, (8, seq_len), device=device)
    runtime(input_ids)
    if runtime.last_stats:
        print(runtime.last_stats.summary())
    print()

    # ------------------------------------------------------------------ #
    # Step 6: Quality spot-check
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("STEP 6: Quality spot-check...")
    print("=" * 70)
    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Explain quantum computing in one sentence:",
        "The largest planet in our solar system is",
    ]

    # Use threshold=0.7 for quality check (moderate aggressiveness)
    tide_config = TIDEConfig(
        checkpoint_interval=checkpoint_interval,
        exit_threshold=0.7,
        min_layers=4,
    )
    runtime = TIDERuntime(model, router_path, config=tide_config, use_cuda_kernels=False)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Vanilla HF
        with torch.no_grad():
            vanilla_logits = model(**inputs).logits
        vanilla_pred = tokenizer.decode(vanilla_logits[0, -1].argmax().item())

        # TIDE
        tide_logits = runtime(inputs.input_ids)
        tide_pred = tokenizer.decode(tide_logits[0, -1].argmax().item())

        match = "MATCH" if vanilla_pred.strip() == tide_pred.strip() else "DIFF"
        print(f"  '{prompt}' -> HF:'{vanilla_pred.strip()}' TIDE:'{tide_pred.strip()}' [{match}]")

    # ------------------------------------------------------------------ #
    # Step 7: Decode-time layer skipping (THE KEY TEST)
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("STEP 7: Decode-time layer skipping (generation)...")
    print("=" * 70)
    gen_prompts = [
        "Explain the theory of relativity in simple terms:",
        "Write a Python function to check if a number is prime:",
        "Solve step by step: If 3x + 7 = 22, what is x?",
    ]
    for threshold in [0.3, 0.5, 0.7]:
        tide_cfg = TIDEConfig(checkpoint_interval=checkpoint_interval, exit_threshold=threshold, min_layers=4)
        runtime = TIDERuntime(model, router_path, config=tide_cfg, use_cuda_kernels=False)
        print(f"\n  --- Threshold={threshold} ---")

        for prompt in gen_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Vanilla HF
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                hf_out = model.generate(
                    inputs.input_ids, max_new_tokens=64,
                    do_sample=False, temperature=None, top_p=None,
                )
            torch.cuda.synchronize()
            hf_time = time.perf_counter() - t0

            # TIDE with layer skipping
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tide_out = runtime.generate(inputs.input_ids, max_new_tokens=64, temperature=0)
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

            print(f"\n  '{prompt[:50]}...'")
            print(f"    HF:   {hf_time:.2f}s | {hf_text[:80]}...")
            print(f"    TIDE: {tide_time:.2f}s | {tide_text[:80]}...")
            print(f"    Speedup: {speedup:.2f}x | Exit rate: {exit_rate:.1%}{exits_info}")

    print()

    # ------------------------------------------------------------------ #
    # Summary report
    # ------------------------------------------------------------------ #
    print()
    print("=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Layers: {n_layers}  |  Checkpoints: {checkpoint_layers}")
    print(f"Calibration: {calibration_samples} samples, convergence_threshold={convergence_threshold}")
    print()
    print(
        f"{'Threshold':<12} {'BS':<6} {'HF tok/s':<14} {'TIDE tok/s':<14} "
        f"{'Speedup':<10} {'Exit Rate':<10}"
    )
    print("-" * 66)
    for threshold in thresholds:
        for bs in batch_sizes:
            r = all_tide_results[threshold][bs]
            print(
                f"{threshold:<12.2f} {bs:<6} {hf_results[bs]:<14,.0f} {r['tps']:<14,.0f} "
                f"{r['speedup']:<10.2f} {r['exit_rate']:<10.1%}"
            )
    print()
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


@app.local_entrypoint()
def main():
    benchmark_8b.remote()
