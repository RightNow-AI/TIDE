"""
TIDE with any HuggingFace model — demonstrates the UniversalAdapter.

The UniversalAdapter auto-probes model structure, so you don't need to write
adapter code. Just point it at any causal LM on HuggingFace.

Usage:
    python examples/any_model.py --model "gpt2"
    python examples/any_model.py --model "EleutherAI/pythia-1.4b"
    python examples/any_model.py --model "microsoft/phi-2"
    python examples/any_model.py --model "tiiuae/falcon-7b"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from TIDE import TIDE, TIDEConfig, calibrate
from TIDE.adapters.auto import get_adapter

def main():
    parser = argparse.ArgumentParser(description="TIDE with any model")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--calibration-samples", type=int, default=100)
    args = parser.parse_args()

    # Load model
    print(f"Loading {args.model}...")
    dtype = torch.float32 if "gpt2" in args.model else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Show what the adapter found
    adapter = get_adapter(model)
    layers = adapter.get_layers(model)
    dim = adapter.get_router_input_dim(model)
    norm = adapter.get_final_norm(model)
    print(f"  Adapter:    {adapter.__class__.__name__}")
    print(f"  Layers:     {len(layers)}")
    print(f"  Hidden dim: {dim}")
    print(f"  Norm:       {norm.__class__.__name__}")

    # Calibrate
    print(f"\nCalibrating ({args.calibration_samples} samples)...")
    config = TIDEConfig(
        calibration_samples=args.calibration_samples,
        checkpoint_interval=4,
    )
    router_path = f"/tmp/tide_{args.model.replace('/', '_')}_router.pt"
    calibrate(model, tokenizer, config=config, save_path=router_path)

    # Run inference
    engine = TIDE(model, router_path, config=TIDEConfig(exit_threshold=0.5))

    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "In machine learning, gradient descent is",
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = engine.generate(inputs.input_ids, max_new_tokens=50, temperature=0)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        stats = engine.last_stats

        print(f"\n  Prompt: {prompt!r}")
        print(f"  Output: {text[:120]}")
        print(f"  Exits:  {stats.exit_rate:.0%} ({stats.total_exited}/{stats.total_tokens})")

if __name__ == "__main__":
    main()
