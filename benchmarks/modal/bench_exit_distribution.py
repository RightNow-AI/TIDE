"""Exit distribution benchmark: per-layer exit histograms."""

import json
import time
from pathlib import Path

import torch


def bench_exit_distribution(
    model_name: str,
    router_path: str,
    batch_sizes: list[int] = [1, 8, 32],
    seq_len: int = 512,
    threshold: float = 0.85,
    n_samples: int = 100,
    output_dir: str = "./results",
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from TIDE.config import TIDEConfig
    from TIDE.runtime import TIDERuntime

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = TIDEConfig(exit_threshold=threshold)
    runtime = TIDERuntime(model, router_path, config=config)

    results = {
        "model": model_name,
        "benchmark": "exit_distribution",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "threshold": threshold,
        "entries": [],
    }

    for bs in batch_sizes:
        layer_totals = {}
        total_tokens = 0
        total_full_depth = 0

        for _ in range(n_samples // bs):
            input_ids = torch.randint(
                0, tokenizer.vocab_size, (bs, seq_len), device=model.device
            )
            runtime(input_ids)

            if runtime.last_stats:
                total_tokens += runtime.last_stats.total_tokens
                total_full_depth += runtime.last_stats.remaining_tokens
                for layer, count in runtime.last_stats.exits_per_layer.items():
                    layer_totals[layer] = layer_totals.get(layer, 0) + count

        histogram = {}
        for layer in sorted(layer_totals.keys()):
            pct = layer_totals[layer] / total_tokens * 100 if total_tokens > 0 else 0
            histogram[str(layer)] = {"count": layer_totals[layer], "pct": round(pct, 2)}

        full_depth_pct = total_full_depth / total_tokens * 100 if total_tokens > 0 else 0

        entry = {
            "batch_size": bs,
            "total_tokens": total_tokens,
            "histogram": histogram,
            "full_depth_tokens": total_full_depth,
            "full_depth_pct": round(full_depth_pct, 2),
        }
        results["entries"].append(entry)
        print(f"  BS={bs}: {len(histogram)} exit layers, {full_depth_pct:.1f}% full depth")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    out_path = f"{output_dir}/{safe_name}_exit_distribution.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
