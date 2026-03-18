"""Latency benchmark: TTFT + ITL at various batch sizes."""

import json
import time
from pathlib import Path

import torch


def bench_latency(
    model_name: str,
    router_path: str,
    batch_sizes: list[int] = [1, 8, 32],
    seq_len: int = 512,
    max_new_tokens: int = 128,
    threshold: float = 0.85,
    n_runs: int = 5,
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
        "benchmark": "latency",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "entries": [],
    }

    for bs in batch_sizes:
        ttft_times = []
        itl_times = []

        for _ in range(n_runs):
            input_ids = torch.randint(
                0, tokenizer.vocab_size, (bs, seq_len), device=model.device
            )

            # TTFT: time to first token (prefill)
            torch.cuda.synchronize()
            start = time.perf_counter()
            logits = runtime(input_ids)
            torch.cuda.synchronize()
            ttft = time.perf_counter() - start
            ttft_times.append(ttft)

            # ITL: inter-token latency (single token decode steps)
            token_times = []
            current = input_ids
            for _ in range(min(max_new_tokens, 10)):
                torch.cuda.synchronize()
                start = time.perf_counter()
                logits = runtime(current)
                next_tok = logits[:, -1:, :].argmax(dim=-1)
                torch.cuda.synchronize()
                token_times.append(time.perf_counter() - start)
                current = torch.cat([current, next_tok], dim=1)

            if token_times:
                itl_times.append(sum(token_times) / len(token_times))

        entry = {
            "batch_size": bs,
            "seq_len": seq_len,
            "threshold": threshold,
            "ttft_ms": sum(ttft_times) / len(ttft_times) * 1000,
            "itl_ms": sum(itl_times) / len(itl_times) * 1000 if itl_times else 0,
        }
        results["entries"].append(entry)
        print(f"  BS={bs}: TTFT={entry['ttft_ms']:.1f}ms  ITL={entry['itl_ms']:.1f}ms")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    out_path = f"{output_dir}/{safe_name}_latency.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
