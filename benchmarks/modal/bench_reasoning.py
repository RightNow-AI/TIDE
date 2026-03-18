"""Reasoning benchmark: R1/QwQ prompts, exit distribution, MATH-500/GPQA."""

import json
import time
from pathlib import Path

import torch


def bench_reasoning(
    model_name: str,
    router_path: str,
    thresholds: list[float] = [0.8, 0.85, 0.9],
    max_new_tokens: int = 2048,
    output_dir: str = "./results",
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from TIDE.config import TIDEConfig
    from TIDE.runtime import TIDERuntime

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = _load_reasoning_prompts()

    results = {
        "model": model_name,
        "benchmark": "reasoning",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "entries": [],
    }

    for threshold in thresholds:
        config = TIDEConfig(exit_threshold=threshold)
        runtime = TIDERuntime(model, router_path, config=config)

        total_tokens = 0
        total_exits = 0
        layer_exit_counts = {}
        generation_times = []

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(model.device)

            torch.cuda.synchronize()
            start = time.perf_counter()
            output = runtime.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0)
            torch.cuda.synchronize()
            gen_time = time.perf_counter() - start
            generation_times.append(gen_time)

            if runtime.last_stats:
                total_tokens += runtime.last_stats.total_tokens
                total_exits += runtime.last_stats.total_exited
                for layer, count in runtime.last_stats.exits_per_layer.items():
                    layer_exit_counts[layer] = layer_exit_counts.get(layer, 0) + count

        entry = {
            "threshold": threshold,
            "avg_generation_time_s": sum(generation_times) / len(generation_times) if generation_times else 0,
            "total_tokens": total_tokens,
            "total_exits": total_exits,
            "exit_rate": total_exits / total_tokens if total_tokens > 0 else 0,
            "exit_distribution": {str(k): v for k, v in sorted(layer_exit_counts.items())},
        }
        results["entries"].append(entry)
        print(f"  thr={threshold}: exit_rate={entry['exit_rate']:.1%} "
              f"avg_time={entry['avg_generation_time_s']:.2f}s")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    out_path = f"{output_dir}/{safe_name}_reasoning.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def _load_reasoning_prompts() -> list[str]:
    """Load reasoning-heavy prompts for benchmarking."""
    return [
        "Solve step by step: What is the sum of all prime numbers less than 100?",
        "Prove that the square root of 2 is irrational.",
        "A train leaves station A at 60mph. Another train leaves station B (300 miles away) at 40mph toward A. When do they meet? Show your work.",
        "Write a Python function to find the longest increasing subsequence. Explain your approach.",
        "What is the probability of getting exactly 3 heads in 10 coin flips? Show the calculation.",
    ]
