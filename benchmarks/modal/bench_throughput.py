"""Throughput benchmark: tokens/sec at various batch sizes comparing baselines."""

import json
import time
from pathlib import Path
from typing import Optional

import torch
import yaml


def bench_throughput(
    model_name: str,
    router_path: str,
    batch_sizes: list[int] = [1, 4, 8, 16, 32, 64, 128],
    seq_len: int = 512,
    thresholds: list[float] = [0.85],
    warmup_iters: int = 3,
    bench_iters: int = 10,
    output_dir: str = "./results",
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from TIDE.config import TIDEConfig
    from TIDE.runtime import TIDERuntime

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    results = {
        "model": model_name,
        "benchmark": "throughput",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "entries": [],
    }

    for threshold in thresholds:
        config = TIDEConfig(exit_threshold=threshold)
        runtime = TIDERuntime(model, router_path, config=config)

        for bs in batch_sizes:
            input_ids = torch.randint(0, tokenizer.vocab_size, (bs, seq_len), device=model.device)

            # TIDE
            for _ in range(warmup_iters):
                runtime(input_ids)
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(bench_iters):
                runtime(input_ids)
            torch.cuda.synchronize()
            tide_elapsed = time.perf_counter() - start
            tide_tps = (bs * seq_len * bench_iters) / tide_elapsed

            # Vanilla HF baseline
            with torch.no_grad():
                for _ in range(warmup_iters):
                    model(input_ids)
                torch.cuda.synchronize()

                start = time.perf_counter()
                for _ in range(bench_iters):
                    model(input_ids)
                torch.cuda.synchronize()
                hf_elapsed = time.perf_counter() - start
                hf_tps = (bs * seq_len * bench_iters) / hf_elapsed

            entry = {
                "batch_size": bs,
                "seq_len": seq_len,
                "threshold": threshold,
                "tide_tokens_per_sec": tide_tps,
                "hf_tokens_per_sec": hf_tps,
                "speedup": tide_tps / hf_tps if hf_tps > 0 else 0,
                "exit_rate": runtime.last_stats.exit_rate if runtime.last_stats else 0,
            }
            results["entries"].append(entry)
            print(f"  BS={bs} thr={threshold}: TIDE={tide_tps:.0f} HF={hf_tps:.0f} "
                  f"({entry['speedup']:.2f}x) exit_rate={entry['exit_rate']:.1%}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    out_path = f"{output_dir}/{safe_name}_throughput.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
