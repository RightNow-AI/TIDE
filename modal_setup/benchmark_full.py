"""
Full TIDE benchmark: calibrate + measure exit rates, latency, and quality on real models.
Runs on Modal A100. Self-contained — no pre-calibrated routers needed.
"""

import modal

from modal_setup.image import build_tide_image
from modal_setup.volumes import VOLUME_MOUNTS

app = modal.App("TIDE-benchmark-full")
tide_image = build_tide_image(include_bench_deps=False)


@app.function(
    image=tide_image,
    gpu="A100",
    volumes=VOLUME_MOUNTS,
    timeout=7200,
)
def benchmark_model(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    calibration_samples: int = 2000,
):
    import time
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from TIDE import TIDE as TIDERuntime, TIDEConfig, calibrate

    print(f"{'='*70}")
    print(f"TIDE BENCHMARK: {model_name}")
    print(f"{'='*70}")

    # ---- Load model ----
    print(f"\nLoading {model_name} (fp16)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        cache_dir="/root/models",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/root/models")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Layers: {model.config.num_hidden_layers}, Hidden: {model.config.hidden_size}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  GPU memory: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

    # ---- Calibrate ----
    safe_name = model_name.replace("/", "_")
    router_path = f"/root/routers/{safe_name}_router.pt"

    print(f"\nCalibrating ({calibration_samples} WikiText samples)...")
    t0 = time.time()
    config = TIDEConfig(
        calibration_samples=calibration_samples,
        checkpoint_interval=4,
        convergence_threshold=0.98,
    )
    ckpt = calibrate(model, tokenizer, config=config, save_path=router_path)
    cal_time = time.time() - t0
    print(f"  Calibrated in {cal_time:.1f}s")
    print(f"  Routers at layers: {sorted(ckpt.routers.keys())}")

    # ---- Real text prompts for evaluation ----
    eval_prompts = [
        "Explain the theory of general relativity in simple terms.",
        "Write a Python function that implements binary search.",
        "What are the main causes of climate change?",
        "Describe the process of photosynthesis step by step.",
        "Compare and contrast TCP and UDP protocols.",
        "What were the main causes of World War I?",
        "Explain how neural networks learn through backpropagation.",
        "What is the difference between a stack and a queue?",
        "Describe the water cycle and its importance to life on Earth.",
        "How does encryption work to protect data?",
        "What are the principles of object-oriented programming?",
        "Explain the concept of supply and demand in economics.",
        "How do vaccines work to prevent disease?",
        "What is the significance of the Pythagorean theorem?",
        "Describe the structure of DNA and its role in genetics.",
        "What are the main differences between Python and JavaScript?",
    ]

    all_results = {
        "model": model_name,
        "layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "gpu": torch.cuda.get_device_name(),
        "calibration_time_s": round(cal_time, 1),
        "calibration_samples": calibration_samples,
    }

    # ==== BENCHMARK 1: Prefill Exit Rates ====
    print(f"\n{'='*70}")
    print("BENCHMARK 1: Prefill Exit Rates (forward pass on real text)")
    print(f"{'='*70}")
    print(f"  {len(eval_prompts)} prompts, sweeping thresholds")
    print(f"\n{'Threshold':>10} {'Exit Rate':>10} {'Tokens':>8} {'Exited':>8} {'Per Layer':>30}")
    print("-" * 70)

    prefill_results = []
    thresholds = [0.95, 0.90, 0.85, 0.80, 0.70, 0.50, 0.30]

    for thresh in thresholds:
        cfg = TIDEConfig(exit_threshold=thresh, min_layers=8)
        engine = TIDERuntime(model, router_path, config=cfg)

        total_tokens = 0
        total_exited = 0
        layer_exits = {}

        for prompt in eval_prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=512).to(model.device)
            engine(inputs.input_ids, attention_mask=inputs.attention_mask)
            s = engine.last_stats
            total_tokens += s.total_tokens
            total_exited += s.total_exited
            for l, c in s.exits_per_layer.items():
                layer_exits[l] = layer_exits.get(l, 0) + c

        rate = total_exited / total_tokens if total_tokens > 0 else 0
        layer_str = " ".join(f"L{l}:{c}" for l, c in sorted(layer_exits.items()))
        print(f"{thresh:>10.2f} {rate:>9.1%} {total_tokens:>8} {total_exited:>8} {layer_str:>30}")

        prefill_results.append({
            "threshold": thresh,
            "exit_rate": round(rate, 4),
            "total_tokens": total_tokens,
            "total_exited": total_exited,
            "layer_exits": {str(k): v for k, v in sorted(layer_exits.items())},
        })

    all_results["prefill_exit_rates"] = prefill_results

    # ==== BENCHMARK 2: Prefill Latency (TIDE vs baseline) ====
    print(f"\n{'='*70}")
    print("BENCHMARK 2: Prefill Latency — Baseline vs TIDE")
    print(f"{'='*70}")

    latency_results = []

    # Baseline (no TIDE)
    test_prompt = "Explain the theory of general relativity in simple terms."
    inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True,
                      max_length=512).to(model.device)

    # Warmup
    for _ in range(3):
        model(inputs.input_ids, attention_mask=inputs.attention_mask)
    torch.cuda.synchronize()

    # Baseline timing
    n_runs = 20
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(inputs.input_ids, attention_mask=inputs.attention_mask)
    torch.cuda.synchronize()
    baseline_ms = (time.perf_counter() - t0) / n_runs * 1000

    print(f"\n  Baseline (vanilla model):  {baseline_ms:.2f}ms / forward pass")

    for thresh in [0.95, 0.85, 0.70, 0.50]:
        cfg = TIDEConfig(exit_threshold=thresh, min_layers=8)
        engine = TIDERuntime(model, router_path, config=cfg)

        # Warmup
        for _ in range(3):
            engine(inputs.input_ids, attention_mask=inputs.attention_mask)
        torch.cuda.synchronize()

        # TIDE timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            engine(inputs.input_ids, attention_mask=inputs.attention_mask)
        torch.cuda.synchronize()
        tide_ms = (time.perf_counter() - t0) / n_runs * 1000

        exit_rate = engine.last_stats.exit_rate if engine.last_stats else 0
        overhead_pct = (tide_ms - baseline_ms) / baseline_ms * 100

        print(f"  TIDE (threshold={thresh}):  {tide_ms:.2f}ms  "
              f"(exit_rate={exit_rate:.0%}, overhead={overhead_pct:+.1f}%)")

        latency_results.append({
            "threshold": thresh,
            "baseline_ms": round(baseline_ms, 2),
            "tide_ms": round(tide_ms, 2),
            "overhead_pct": round(overhead_pct, 1),
            "exit_rate": round(exit_rate, 4),
        })

    all_results["prefill_latency"] = latency_results

    # ==== BENCHMARK 3: Generation Quality ====
    print(f"\n{'='*70}")
    print("BENCHMARK 3: Generation Quality (100 tokens, temp=0)")
    print(f"{'='*70}")

    gen_prompt = "Explain how neural networks learn through backpropagation:"
    gen_inputs = tokenizer(gen_prompt, return_tensors="pt").to(model.device)

    gen_results = []

    for thresh in [1.0, 0.85, 0.70, 0.50]:
        cfg = TIDEConfig(exit_threshold=thresh, min_layers=8)
        engine = TIDERuntime(model, router_path, config=cfg)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = engine.generate(gen_inputs.input_ids, max_new_tokens=100, temperature=0)
        torch.cuda.synchronize()
        gen_time = time.perf_counter() - t0

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_ids = output[0][gen_inputs.input_ids.shape[1]:]
        n_generated = len(generated_ids)
        unique_tokens = len(set(generated_ids.tolist()))
        stats = engine.last_stats

        label = "baseline" if thresh == 1.0 else f"thresh={thresh}"
        print(f"\n  [{label}] {gen_time:.1f}s, {n_generated} tokens, "
              f"exit_rate={stats.exit_rate:.0%}, unique={unique_tokens}")
        print(f"  {text[:200]}")

        gen_results.append({
            "threshold": thresh,
            "time_s": round(gen_time, 2),
            "n_generated": n_generated,
            "unique_tokens": unique_tokens,
            "exit_rate": round(stats.exit_rate, 4),
            "exits_per_layer": {str(k): v for k, v in stats.exits_per_layer.items()},
            "text_preview": text[:300],
        })

    all_results["generation_quality"] = gen_results

    # ==== BENCHMARK 4: Batch prefill throughput ====
    print(f"\n{'='*70}")
    print("BENCHMARK 4: Batch Prefill Throughput (tokens/sec)")
    print(f"{'='*70}")

    throughput_results = []

    for batch_size in [1, 4, 8]:
        # Create batch of real prompts
        batch_prompts = (eval_prompts * 4)[:batch_size]
        batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                                truncation=True, max_length=256).to(model.device)
        n_tokens = batch_inputs.input_ids.numel()

        # Baseline
        for _ in range(3):
            model(**batch_inputs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            model(**batch_inputs)
        torch.cuda.synchronize()
        baseline_tps = n_tokens * 10 / (time.perf_counter() - t0)

        # TIDE @ 0.85
        cfg = TIDEConfig(exit_threshold=0.85, min_layers=8)
        engine = TIDERuntime(model, router_path, config=cfg)

        for _ in range(3):
            engine(batch_inputs.input_ids, attention_mask=batch_inputs.attention_mask)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            engine(batch_inputs.input_ids, attention_mask=batch_inputs.attention_mask)
        torch.cuda.synchronize()
        tide_tps = n_tokens * 10 / (time.perf_counter() - t0)

        exit_rate = engine.last_stats.exit_rate if engine.last_stats else 0

        print(f"  BS={batch_size}: baseline={baseline_tps:,.0f} tok/s, "
              f"TIDE={tide_tps:,.0f} tok/s ({exit_rate:.0%} exits)")

        throughput_results.append({
            "batch_size": batch_size,
            "baseline_tokens_per_sec": round(baseline_tps),
            "tide_tokens_per_sec": round(tide_tps),
            "exit_rate": round(exit_rate, 4),
        })

    all_results["batch_throughput"] = throughput_results

    # ==== Save results ====
    save_path = f"/root/results/{safe_name}_full_benchmark.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(json.dumps(all_results, indent=2))

    return all_results


@app.local_entrypoint()
def main():
    results = benchmark_model.remote()
    print("\nBenchmark complete.")
