"""TIDE experiment: measure exit rates, convergence, and generation quality."""

import modal

from modal_setup.image import build_tide_image
from modal_setup.volumes import VOLUME_MOUNTS

app = modal.App("TIDE-experiment")
tide_image = build_tide_image(include_bench_deps=False)


@app.function(
    image=tide_image,
    gpu="A10G",
    volumes=VOLUME_MOUNTS,
    timeout=3600,
)
def run_experiment():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from TIDE import TIDE as TIDERuntime, TIDEConfig, calibrate

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        cache_dir="/root/models",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/root/models")
    tokenizer.pad_token = tokenizer.eos_token

    # ---- Experiment 1: Calibration with full samples ----
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Calibration Convergence (2000 samples)")
    print("=" * 70)

    config = TIDEConfig(
        calibration_samples=2000,
        checkpoint_interval=4,
        convergence_threshold=0.98,
    )
    ckpt = calibrate(model, tokenizer, config=config, save_path="/tmp/exp_router.pt")
    print(f"Routers trained: {len(ckpt.routers)} at layers {sorted(ckpt.routers.keys())}")

    # ---- Experiment 2: Forward pass exit rates (prefill) ----
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Prefill Exit Rates (forward pass)")
    print("=" * 70)

    prompts = [
        "The theory of general relativity states that",
        "In Python, you can create a list comprehension by",
        "The three branches of the United States government are",
        "Machine learning models learn patterns from data by",
        "To make a good cup of coffee, you need to",
        "The Pythagorean theorem states that in a right triangle",
        "Climate change is caused primarily by",
        "The Roman Empire fell due to a combination of",
    ]

    thresholds = [0.95, 0.90, 0.85, 0.80, 0.70, 0.50]

    print(f"\n{'Threshold':>10} {'Exit Rate':>10} {'Exits @ Layer':>40}")
    print("-" * 65)

    for thresh in thresholds:
        cfg = TIDEConfig(exit_threshold=thresh, min_layers=4)
        engine = TIDERuntime(model, "/tmp/exp_router.pt", config=cfg)

        total_tokens = 0
        total_exited = 0
        layer_exits = {}

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            engine(inputs.input_ids)
            stats = engine.last_stats
            total_tokens += stats.total_tokens
            total_exited += stats.total_exited
            for l, c in stats.exits_per_layer.items():
                layer_exits[l] = layer_exits.get(l, 0) + c

        rate = total_exited / total_tokens if total_tokens > 0 else 0
        layer_str = ", ".join(f"L{l}:{c}" for l, c in sorted(layer_exits.items()))
        print(f"{thresh:>10.2f} {rate:>9.0%} {layer_str:>40}")

    # ---- Experiment 3: Generation quality at different thresholds ----
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Generation Quality vs Exit Rate")
    print("=" * 70)

    gen_prompt = "Explain how neural networks learn:"
    inputs = tokenizer(gen_prompt, return_tensors="pt").to(model.device)

    results = {}
    for thresh in [1.0, 0.85, 0.70, 0.50]:
        cfg = TIDEConfig(exit_threshold=thresh, min_layers=4)
        engine = TIDERuntime(model, "/tmp/exp_router.pt", config=cfg)

        output = engine.generate(inputs.input_ids, max_new_tokens=100, temperature=0)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        stats = engine.last_stats

        generated_ids = output[0][inputs.input_ids.shape[1]:]
        unique = len(set(generated_ids.tolist()))

        label = "baseline (no exits)" if thresh == 1.0 else f"threshold={thresh}"
        results[thresh] = {
            "exit_rate": stats.exit_rate,
            "unique_tokens": unique,
            "text": text,
            "exits_per_layer": dict(stats.exits_per_layer),
        }

        print(f"\n  [{label}]")
        print(f"  Exit rate: {stats.exit_rate:.0%}, Unique tokens: {unique}")
        print(f"  Layer exits: {dict(stats.exits_per_layer)}")
        print(f"  Output: {text[:200]}")

    # ---- Experiment 4: GPT-2 (different architecture) ----
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: GPT-2 (Pattern B architecture)")
    print("=" * 70)

    del model
    torch.cuda.empty_cache()

    gpt2 = AutoModelForCausalLM.from_pretrained(
        "gpt2", cache_dir="/root/models", device_map="auto",
    )
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2", cache_dir="/root/models")
    gpt2_tok.pad_token = gpt2_tok.eos_token

    gpt2_config = TIDEConfig(
        calibration_samples=500,
        checkpoint_interval=4,
        convergence_threshold=0.98,
    )
    calibrate(gpt2, gpt2_tok, config=gpt2_config, save_path="/tmp/gpt2_router.pt")

    for thresh in [0.85, 0.70, 0.50]:
        cfg = TIDEConfig(exit_threshold=thresh, min_layers=4)
        engine = TIDERuntime(gpt2, "/tmp/gpt2_router.pt", config=cfg)

        gpt2_prompts = [
            "The meaning of life is",
            "In the year 2050, technology will",
            "The best programming language is",
            "Once upon a time in a land far away",
        ]

        total_tokens = 0
        total_exited = 0
        for p in gpt2_prompts:
            inp = gpt2_tok(p, return_tensors="pt").to(gpt2.device)
            engine(inp.input_ids)
            s = engine.last_stats
            total_tokens += s.total_tokens
            total_exited += s.total_exited

        rate = total_exited / total_tokens
        print(f"  GPT-2 @ threshold={thresh}: {rate:.0%} exit rate ({total_exited}/{total_tokens} tokens)")

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)

    return results


@app.local_entrypoint()
def main():
    results = run_experiment.remote()
    print("\nExperiment complete.")
