"""
Integrate TIDE with existing HuggingFace code.

Shows how to drop TIDE into code that already uses transformers,
with minimal changes. The key pattern:

    BEFORE:  logits = model(input_ids).logits
    AFTER:   logits = engine(input_ids)  # same shape, fewer compute

Usage:
    python examples/huggingface_pipeline.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from TIDE import TIDE, TIDEConfig, calibrate

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ROUTER_PATH = "/tmp/tide_hf_router.pt"

def main():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device

    # ---- Calibrate once ----
    import os
    if not os.path.exists(ROUTER_PATH):
        print("Calibrating (one-time)...")
        calibrate(model, tokenizer, save_path=ROUTER_PATH,
                  config=TIDEConfig(calibration_samples=100))

    engine = TIDE(model, ROUTER_PATH, config=TIDEConfig(exit_threshold=0.85))

    # ================================================================
    # Pattern 1: Drop-in replacement for model.generate()
    # ================================================================
    print("\n--- Pattern 1: Generation ---")
    prompt = "The key insight of attention mechanisms is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Before: output_ids = model.generate(inputs.input_ids, max_new_tokens=64)
    # After:
    output_ids = engine.generate(inputs.input_ids, max_new_tokens=64, temperature=0)
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    print(f"  Exit rate: {engine.last_stats.exit_rate:.0%}")

    # ================================================================
    # Pattern 2: Scoring / classification (forward pass)
    # ================================================================
    print("\n--- Pattern 2: Forward pass (logits) ---")
    texts = ["This movie was great!", "This movie was terrible!"]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)

    # Before: logits = model(inputs.input_ids).logits
    # After:
    logits = engine(inputs.input_ids, attention_mask=inputs.attention_mask)
    print(f"  Logit shapes: {logits.shape}")
    print(f"  Exit rate: {engine.last_stats.exit_rate:.0%}")

    # ================================================================
    # Pattern 3: Batch generation loop
    # ================================================================
    print("\n--- Pattern 3: Batch processing ---")
    prompts = [
        "Python is",
        "Machine learning",
        "The internet was",
    ]
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        out = engine.generate(ids, max_new_tokens=30, temperature=0)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        rate = engine.last_stats.exit_rate
        print(f"  [{rate:3.0%} exits] {text[:80]}")

if __name__ == "__main__":
    main()
