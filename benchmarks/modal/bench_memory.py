"""Memory benchmark: peak GPU memory vs baselines."""

import json
import time
from pathlib import Path

import torch


def bench_memory(
    model_name: str,
    router_path: str,
    batch_sizes: list[int] = [1, 8, 32],
    seq_len: int = 512,
    threshold: float = 0.85,
    output_dir: str = "./results",
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from TIDE.config import TIDEConfig
    from TIDE.runtime import TIDERuntime

    results = {
        "model": model_name,
        "benchmark": "memory",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "entries": [],
    }

    for bs in batch_sizes:
        # Measure baseline HF memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        input_ids = torch.randint(0, tokenizer.vocab_size, (bs, seq_len), device=model.device)

        with torch.no_grad():
            model(input_ids)
        torch.cuda.synchronize()
        hf_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Measure TIDE memory
        torch.cuda.reset_peak_memory_stats()

        config = TIDEConfig(exit_threshold=threshold)
        runtime = TIDERuntime(model, router_path, config=config)
        runtime(input_ids)
        torch.cuda.synchronize()
        tide_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        entry = {
            "batch_size": bs,
            "seq_len": seq_len,
            "hf_peak_mb": hf_peak_mb,
            "tide_peak_mb": tide_peak_mb,
            "overhead_mb": tide_peak_mb - hf_peak_mb,
            "overhead_pct": (tide_peak_mb - hf_peak_mb) / hf_peak_mb * 100 if hf_peak_mb > 0 else 0,
        }
        results["entries"].append(entry)
        print(f"  BS={bs}: HF={hf_peak_mb:.0f}MB TIDE={tide_peak_mb:.0f}MB "
              f"overhead={entry['overhead_pct']:+.1f}%")

        del runtime, model
        torch.cuda.empty_cache()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    out_path = f"{output_dir}/{safe_name}_memory.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
