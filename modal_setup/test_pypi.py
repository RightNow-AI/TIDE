"""Test that tide-inference installs and works from PyPI."""
import modal

app = modal.App("tide-pypi-test")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("torch>=2.5.0", "numpy", "setuptools>=68.0", "wheel", "ninja")
    .pip_install("transformers>=4.48.0", "datasets", "pyyaml", "accelerate", "sentencepiece", "protobuf")
    .pip_install("tide-inference==0.2.1", gpu="A10G")
)

@app.function(image=image, gpu="A10G", timeout=600)
def test():
    import torch
    import TIDE
    print(f"TIDE {TIDE.__version__} installed from PyPI")
    print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name()}")
    print(f"Exports: {TIDE.__all__}")

    # Test config
    cfg = TIDE.TIDEConfig(exit_threshold=0.85)
    print(f"Config OK: threshold={cfg.exit_threshold}")

    # Test universal adapter on GPT-2
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    from TIDE.adapters.auto import get_adapter
    adapter = get_adapter(model)
    print(f"GPT-2 adapter: {adapter.__class__.__name__}, {len(adapter.get_layers(model))} layers")

    # Test calibration
    ckpt = TIDE.calibrate(model, tok, save_path="/tmp/test_router.pt",
                          config=TIDE.TIDEConfig(calibration_samples=20, checkpoint_interval=4))
    print(f"Calibrated: {len(ckpt.routers)} routers")

    # Test inference
    engine = TIDE.TIDE(model, "/tmp/test_router.pt",
                       config=TIDE.TIDEConfig(exit_threshold=0.5, min_layers=0))
    inputs = tok("Hello world", return_tensors="pt")
    logits = engine(inputs.input_ids)
    print(f"Forward pass: {logits.shape}")
    print(f"Exit stats: {engine.last_stats.summary()}")

    print("\nALL TESTS PASSED — PyPI package works!")

@app.local_entrypoint()
def main():
    test.remote()
