"""
Threshold tuning — find the best exit_threshold for your quality/speed tradeoff.

Sweeps threshold from 0.3 (aggressive, more exits) to 0.95 (conservative, fewer exits)
and shows the quality/exit-rate tradeoff on a sample prompt set.

Usage:
    python examples/tune_threshold.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from TIDE import TIDE, TIDEConfig, calibrate

EVAL_PROMPTS = [
    "What is photosynthesis?",
    "Write a Python function to sort a list.",
    "The three laws of thermodynamics are:",
    "Explain the difference between TCP and UDP.",
    "Translate to French: The weather is nice today.",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--router-path", default="/tmp/tide_tune_router.pt")
    parser.add_argument("--calibration-samples", type=int, default=200)
    args = parser.parse_args()

    # Load and calibrate
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import os
    if not os.path.exists(args.router_path):
        print("Calibrating...")
        config = TIDEConfig(calibration_samples=args.calibration_samples)
        calibrate(model, tokenizer, config=config, save_path=args.router_path)

    # Baseline: no TIDE (threshold=1.0 means no exits)
    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP")
    print("=" * 70)
    print(f"{'Threshold':>10} {'Exit Rate':>10} {'Avg Unique Tokens':>18} {'Quality':>10}")
    print("-" * 70)

    thresholds = [1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.50, 0.30]

    for thresh in thresholds:
        config = TIDEConfig(exit_threshold=thresh, min_layers=4)
        engine = TIDE(model, args.router_path, config=config, use_cuda_kernels=False)

        total_exit_rate = 0
        total_unique = 0

        for prompt in EVAL_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output = engine.generate(inputs.input_ids, max_new_tokens=50, temperature=0)
            generated = output[0][inputs.input_ids.shape[1]:]
            unique = len(set(generated.tolist()))
            total_unique += unique
            total_exit_rate += engine.last_stats.exit_rate

        avg_exit = total_exit_rate / len(EVAL_PROMPTS)
        avg_unique = total_unique / len(EVAL_PROMPTS)

        # Simple quality heuristic: more unique tokens = less degenerate
        quality = "baseline" if thresh == 1.0 else ("good" if avg_unique > 15 else "degraded")

        label = f"{thresh:.2f}"
        if thresh == 1.0:
            label = "1.00 (off)"
        print(f"{label:>10} {avg_exit:>9.0%} {avg_unique:>17.0f} {quality:>10}")

    print("-" * 70)
    print("Recommendation: start with 0.85, lower to 0.70 for more speed.")
    print("If quality degrades, raise threshold or increase min_layers.\n")

if __name__ == "__main__":
    main()
