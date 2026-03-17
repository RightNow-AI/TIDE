"""TIDE-ci: CI smoke test app on Modal."""

import modal

from modal_setup.image import build_tide_image
from modal_setup.volumes import VOLUME_MOUNTS

app = modal.App("TIDE-ci")
tide_image = build_tide_image(include_bench_deps=False)


@app.function(
    image=tide_image,
    gpu="A10G",
    volumes=VOLUME_MOUNTS,
    timeout=1800,
)
def smoke_test():
    """Phase 1+4: Compile check + kernel numerical equivalence (fp32/fp16/bf16)."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "/root/TIDE/tests/", "-v", "--tb=short",
         "--import-mode=importlib"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    passed = result.returncode == 0
    print(f"\n{'PASSED' if passed else 'FAILED'} (exit code {result.returncode})")
    return {"phase": "1+4", "passed": passed}


@app.function(
    image=tide_image,
    gpu="A10G",
    volumes=VOLUME_MOUNTS,
    timeout=1800,
)
def test_universal_adapter():
    """Phase 2: Verify UniversalAdapter probes GPT-2 (Pattern B) and TinyLlama (Pattern A)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from TIDE.adapters.auto import get_adapter
    from TIDE.adapters.universal import UniversalAdapter
    from TIDE.config import TIDEConfig
    from TIDE.calibrate import calibrate

    results = {}

    # --- GPT-2 (Pattern B: transformer.h / transformer.ln_f / transformer.wte) ---
    print("=" * 60)
    print("Testing UniversalAdapter on GPT-2 (Pattern B)")
    print("=" * 60)

    gpt2 = AutoModelForCausalLM.from_pretrained(
        "gpt2", cache_dir="/root/models",
    )
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2", cache_dir="/root/models")
    gpt2_tok.pad_token = gpt2_tok.eos_token

    adapter = get_adapter(gpt2)
    assert isinstance(adapter, UniversalAdapter), f"Expected UniversalAdapter, got {type(adapter)}"

    layers = adapter.get_layers(gpt2)
    norm = adapter.get_final_norm(gpt2)
    head = adapter.get_lm_head(gpt2)
    emb = adapter.get_embedding(gpt2)
    dim = adapter.get_router_input_dim(gpt2)

    print(f"  layers: {len(layers)}")
    print(f"  norm: {norm.__class__.__name__}")
    print(f"  head: {head.__class__.__name__} -> {head.out_features}")
    print(f"  embedding: {emb.__class__.__name__}")
    print(f"  hidden_dim: {dim}")

    assert len(layers) == 12, f"GPT-2 should have 12 layers, got {len(layers)}"
    assert dim == 768, f"GPT-2 hidden_dim should be 768, got {dim}"

    # Quick calibration on GPT-2
    config = TIDEConfig(
        calibration_samples=20,
        checkpoint_interval=4,
        min_layers=0,
        router_bottleneck_dim=64,
    )
    ckpt = calibrate(gpt2, gpt2_tok, config=config, save_path="/tmp/gpt2_router.pt")
    print(f"  calibrated: {len(ckpt.routers)} routers")

    results["gpt2"] = {
        "passed": True,
        "n_layers": len(layers),
        "hidden_dim": dim,
        "n_routers": len(ckpt.routers),
    }

    del gpt2, gpt2_tok
    torch.cuda.empty_cache()

    # --- TinyLlama (Pattern A: model.layers / model.norm / model.embed_tokens) ---
    print("\n" + "=" * 60)
    print("Testing UniversalAdapter on TinyLlama (Pattern A)")
    print("=" * 60)

    tlm = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/root/models",
    )
    tlm_tok = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", cache_dir="/root/models"
    )

    # TinyLlama is LlamaForCausalLM — it hits the registry, not UniversalAdapter.
    # Force universal by passing to UniversalAdapter.probe directly.
    ua = UniversalAdapter.probe(tlm)

    layers = ua.get_layers(tlm)
    norm = ua.get_final_norm(tlm)
    head = ua.get_lm_head(tlm)
    dim = ua.get_router_input_dim(tlm)

    print(f"  layers: {len(layers)}")
    print(f"  norm: {norm.__class__.__name__}")
    print(f"  head: {head.__class__.__name__} -> {head.out_features}")
    print(f"  hidden_dim: {dim}")

    assert len(layers) == 22, f"TinyLlama should have 22 layers, got {len(layers)}"
    assert dim == 2048, f"TinyLlama hidden_dim should be 2048, got {dim}"

    results["tinyllama"] = {
        "passed": True,
        "n_layers": len(layers),
        "hidden_dim": dim,
    }

    print("\n" + "=" * 60)
    print("Phase 2: ALL PASSED")
    print("=" * 60)
    return {"phase": "2", "passed": True, "results": results}


@app.function(
    image=tide_image,
    gpu="A10G",
    volumes=VOLUME_MOUNTS,
    timeout=2400,
)
def test_kv_cache_generation():
    """Phase 3: Generate 100 tokens with TIDE, verify KV cache consistency and output coherence."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from TIDE.config import TIDEConfig
    from TIDE.calibrate import calibrate
    from TIDE.runtime import TIDERuntime

    print("=" * 60)
    print("Phase 3: KV Cache Fix — Generation Test")
    print("=" * 60)

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/root/models",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/root/models")
    tokenizer.pad_token = tokenizer.eos_token

    # Calibrate with low samples for speed
    config = TIDEConfig(
        calibration_samples=50,
        checkpoint_interval=4,
        min_layers=4,
        kv_cache_strategy="zero_pad",
    )
    calibrate(model, tokenizer, config=config, save_path="/tmp/kv_router.pt")

    # Generate with low threshold to force many early exits
    gen_config = TIDEConfig(
        exit_threshold=0.3,
        min_layers=4,
        checkpoint_interval=4,
        kv_cache_strategy="zero_pad",
    )
    runtime = TIDERuntime(model, "/tmp/kv_router.pt", config=gen_config)

    prompt = "The theory of general relativity, proposed by Albert Einstein,"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"  prompt: {prompt!r}")
    print(f"  threshold: {gen_config.exit_threshold}")
    print(f"  generating 100 tokens...")

    # This is the critical test — previously would crash or produce garbage
    # due to KV cache shape mismatches on skipped layers
    output_ids = runtime.generate(
        inputs.input_ids,
        max_new_tokens=100,
        temperature=0,
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    stats = runtime.last_stats

    print(f"\n  output ({len(output_ids[0])} tokens):")
    print(f"  {output_text[:300]}")
    print(f"\n  stats: {stats.summary()}")

    # Verify stats are sane
    assert stats.total_tokens > 0, "Should have generated tokens"
    total_accounted = stats.total_exited + stats.remaining_tokens
    assert total_accounted == stats.total_tokens, (
        f"Token accounting mismatch: {stats.total_exited} exited + "
        f"{stats.remaining_tokens} remaining != {stats.total_tokens} total"
    )

    # Verify output isn't degenerate (not all same token, reasonable length)
    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    unique_tokens = len(set(generated_ids.tolist()))
    print(f"  unique tokens in generation: {unique_tokens}")
    assert unique_tokens > 3, f"Output too repetitive: only {unique_tokens} unique tokens"

    exit_rate = stats.exit_rate
    print(f"  exit rate: {exit_rate:.1%}")

    results = {
        "passed": True,
        "n_generated": len(generated_ids),
        "unique_tokens": unique_tokens,
        "exit_rate": exit_rate,
        "exits_per_layer": dict(stats.exits_per_layer),
        "output_preview": output_text[:200],
    }

    print("\n" + "=" * 60)
    print("Phase 3: PASSED")
    print("=" * 60)
    return {"phase": "3", "passed": True, "results": results}


@app.function(
    image=tide_image,
    gpu="A10G",
    volumes=VOLUME_MOUNTS,
    timeout=1800,
)
def integration_test(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Quick integration test with a small model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from TIDE.config import TIDEConfig
    from TIDE.calibrate import calibrate
    from TIDE.runtime import TIDERuntime

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        cache_dir="/root/models",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/root/models")

    config = TIDEConfig(calibration_samples=50, checkpoint_interval=4)
    checkpoint = calibrate(model, tokenizer, config=config, save_path="/tmp/ci_router.pt")

    runtime = TIDERuntime(model, "/tmp/ci_router.pt", config=TIDEConfig())
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
    logits = runtime(inputs.input_ids)

    return {
        "passed": logits.shape[0] == 1 and logits.shape[2] == model.config.vocab_size,
        "output_shape": list(logits.shape),
        "exit_rate": runtime.last_stats.exit_rate if runtime.last_stats else 0,
    }


@app.local_entrypoint()
def main():
    import json

    print("=" * 60)
    print("TIDE Universal CI — All Phases")
    print("=" * 60)

    # Phase 1+4: Compile + kernel tests (fp32/fp16/bf16)
    print("\n>>> Phase 1+4: Compile + Kernel Tests")
    r1 = smoke_test.remote()
    print(f"    Result: {'PASSED' if r1['passed'] else 'FAILED'}")

    # Phase 2: Universal adapter on GPT-2 + TinyLlama
    print("\n>>> Phase 2: Universal Adapter")
    r2 = test_universal_adapter.remote()
    print(f"    Result: {'PASSED' if r2['passed'] else 'FAILED'}")
    if "results" in r2:
        for name, info in r2["results"].items():
            print(f"      {name}: {info}")

    # Phase 3: KV cache generation test
    print("\n>>> Phase 3: KV Cache Generation")
    r3 = test_kv_cache_generation.remote()
    print(f"    Result: {'PASSED' if r3['passed'] else 'FAILED'}")
    if "results" in r3:
        print(f"      exit_rate: {r3['results'].get('exit_rate', 'N/A')}")
        print(f"      generated: {r3['results'].get('n_generated', 'N/A')} tokens")
        print(f"      preview: {r3['results'].get('output_preview', '')[:100]}")

    # Summary
    all_passed = all(r["passed"] for r in [r1, r2, r3])
    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 60)
