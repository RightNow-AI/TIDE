"""Quality benchmark: MMLU/HumanEval/GSM8K/MATH/ARC delta at various thresholds."""

import json
import time
from pathlib import Path

import torch


def bench_quality(
    model_name: str,
    router_path: str,
    thresholds: list[float] = [0.7, 0.8, 0.85, 0.9, 0.95],
    dataset_name: str = "mmlu",
    n_samples: int = 500,
    output_dir: str = "./results",
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from TIDE.config import TIDEConfig
    from TIDE.runtime import TIDERuntime

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    samples = _load_eval_samples(dataset_name, n_samples)

    results = {
        "model": model_name,
        "benchmark": "quality",
        "dataset": dataset_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "entries": [],
    }

    # Baseline: vanilla model
    baseline_correct = _evaluate_model(model, tokenizer, samples, use_tide=False)
    baseline_acc = baseline_correct / len(samples) if samples else 0

    for threshold in thresholds:
        config = TIDEConfig(exit_threshold=threshold)
        runtime = TIDERuntime(model, router_path, config=config)

        tide_correct = _evaluate_model_tide(runtime, tokenizer, samples)
        tide_acc = tide_correct / len(samples) if samples else 0

        entry = {
            "threshold": threshold,
            "baseline_accuracy": baseline_acc,
            "tide_accuracy": tide_acc,
            "delta": tide_acc - baseline_acc,
            "exit_rate": runtime.last_stats.exit_rate if runtime.last_stats else 0,
        }
        results["entries"].append(entry)
        print(f"  thr={threshold}: baseline={baseline_acc:.3f} tide={tide_acc:.3f} "
              f"delta={entry['delta']:+.3f} exit_rate={entry['exit_rate']:.1%}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    out_path = f"{output_dir}/{safe_name}_quality_{dataset_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def _load_eval_samples(dataset_name: str, n_samples: int) -> list[dict]:
    """Load evaluation samples. Returns list of {prompt, answer} dicts."""
    try:
        from datasets import load_dataset

        if dataset_name == "mmlu":
            ds = load_dataset("cais/mmlu", "all", split="test", streaming=True)
        elif dataset_name == "gsm8k":
            ds = load_dataset("openai/gsm8k", "main", split="test", streaming=True)
        elif dataset_name == "arc_challenge":
            ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", streaming=True)
        else:
            return []

        samples = []
        for item in ds:
            sample = {"prompt": str(item.get("question", "")), "answer": str(item.get("answer", ""))}
            samples.append(sample)
            if len(samples) >= n_samples:
                break
        return samples
    except Exception:
        return []


@torch.no_grad()
def _evaluate_model(model, tokenizer, samples, use_tide=False):
    correct = 0
    for sample in samples:
        inputs = tokenizer(sample["prompt"], return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        pred = tokenizer.decode(logits[0, -1].argmax().item())
        if pred.strip().lower() == sample["answer"].strip().lower():
            correct += 1
    return correct


@torch.no_grad()
def _evaluate_model_tide(runtime, tokenizer, samples):
    correct = 0
    for sample in samples:
        inputs = tokenizer(sample["prompt"], return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(next(runtime.model.parameters()).device)
        logits = runtime(input_ids)
        pred = tokenizer.decode(logits[0, -1].argmax().item())
        if pred.strip().lower() == sample["answer"].strip().lower():
            correct += 1
    return correct
