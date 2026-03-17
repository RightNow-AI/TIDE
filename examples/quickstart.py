"""
TIDE Quickstart — Calibrate and run inference in ~10 lines.

Usage:
    python examples/quickstart.py
    python examples/quickstart.py --model "mistralai/Mistral-7B-Instruct-v0.3"
    python examples/quickstart.py --model "Qwen/Qwen2.5-7B-Instruct" --threshold 0.7
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from TIDE import TIDE, TIDEConfig, calibrate

def main():
    parser = argparse.ArgumentParser(description="TIDE quickstart")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--prompt", default="Explain how transformers work in simple terms:")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--calibration-samples", type=int, default=200)
    parser.add_argument("--router-path", default="router.pt")
    args = parser.parse_args()

    # ---- Step 1: Load model ----
    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Step 2: Calibrate routers (skip if already exists) ----
    import os
    if not os.path.exists(args.router_path):
        print(f"Calibrating routers ({args.calibration_samples} samples)...")
        config = TIDEConfig(calibration_samples=args.calibration_samples)
        calibrate(model, tokenizer, config=config, save_path=args.router_path)
        print(f"Saved to {args.router_path}")
    else:
        print(f"Using existing routers: {args.router_path}")

    # ---- Step 3: Wrap model with TIDE ----
    config = TIDEConfig(exit_threshold=args.threshold)
    engine = TIDE(model, router_path=args.router_path, config=config)

    # ---- Step 4: Generate ----
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    output = engine.generate(inputs.input_ids, max_new_tokens=args.max_tokens, temperature=0)
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"\n{'='*60}")
    print(text)
    print(f"{'='*60}")
    print(f"\n{engine.last_stats.summary()}")

if __name__ == "__main__":
    main()
