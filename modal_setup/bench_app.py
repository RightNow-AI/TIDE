"""TIDE-bench: Benchmark app on Modal."""

import modal

from modal_setup.image import build_tide_image
from modal_setup.volumes import VOLUME_MOUNTS

app = modal.App("TIDE-bench")
tide_image = build_tide_image(include_bench_deps=True)

GPU_MAP = {
    "small": "A100",      # <=13B models
    "medium": "H100",     # 13B-34B
    "large": "H100:2",    # 34B-70B
    "xlarge": "H100:4",   # 70B+
}


@app.function(
    image=tide_image,
    gpu="H100",
    volumes=VOLUME_MOUNTS,
    timeout=7200,
    ephemeral_disk=200 * 1024,
)
def run_benchmark(
    model_name: str,
    benchmark_type: str = "throughput",
    batch_sizes: list[int] = [1, 4, 8, 16, 32],
    thresholds: list[float] = [0.7, 0.8, 0.85, 0.9, 0.95],
):
    """Run a specific benchmark for a model."""
    import json
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from TIDE.config import TIDEConfig
    from TIDE.runtime import TIDERuntime

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/root/models",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/root/models")

    safe_name = model_name.replace("/", "_")
    router_path = f"/root/routers/{safe_name}_router.pt"

    results = {
        "model": model_name,
        "benchmark": benchmark_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [],
    }

    if benchmark_type == "throughput":
        for bs in batch_sizes:
            for threshold in thresholds:
                config = TIDEConfig(exit_threshold=threshold)
                runtime = TIDERuntime(model, router_path, config=config)

                input_ids = torch.randint(
                    0, tokenizer.vocab_size, (bs, 512), device=model.device
                )

                # Warmup
                for _ in range(3):
                    runtime(input_ids)
                torch.cuda.synchronize()

                # Benchmark
                start = time.perf_counter()
                n_iters = 10
                for _ in range(n_iters):
                    runtime(input_ids)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                tokens_per_sec = (bs * 512 * n_iters) / elapsed
                results["results"].append({
                    "batch_size": bs,
                    "threshold": threshold,
                    "tokens_per_sec": tokens_per_sec,
                    "exit_rate": runtime.last_stats.exit_rate if runtime.last_stats else 0,
                })

    save_path = f"/root/results/{safe_name}_{benchmark_type}.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
