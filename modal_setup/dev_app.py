"""TIDE-dev: Development and calibration app on Modal."""

import modal

from modal_setup.image import build_tide_image
from modal_setup.volumes import VOLUME_MOUNTS

app = modal.App("TIDE-dev")
tide_image = build_tide_image(include_bench_deps=False)


@app.function(
    image=tide_image,
    gpu="A100",
    volumes=VOLUME_MOUNTS,
    timeout=3600,
)
def calibrate_model(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    num_samples: int = 2000,
    checkpoint_interval: int = 4,
    convergence_threshold: float = 0.98,
):
    """Calibrate TIDE routers for a model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from TIDE.config import TIDEConfig
    from TIDE.calibrate import calibrate

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/root/models",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/root/models")

    config = TIDEConfig(
        checkpoint_interval=checkpoint_interval,
        calibration_samples=num_samples,
        convergence_threshold=convergence_threshold,
    )

    safe_name = model_name.replace("/", "_")
    save_path = f"/root/routers/{safe_name}_router.pt"
    calibrate(model, tokenizer, config=config, save_path=save_path)
    return save_path


@app.function(
    image=tide_image,
    gpu="A100",
    volumes=VOLUME_MOUNTS,
    timeout=3600,
)
def test_inference(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    prompt: str = "Explain quantum computing in simple terms:",
    max_new_tokens: int = 256,
):
    """Test TIDE inference on a model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from TIDE.config import TIDEConfig
    from TIDE.runtime import TIDERuntime

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/root/models",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/root/models")

    safe_name = model_name.replace("/", "_")
    router_path = f"/root/routers/{safe_name}_router.pt"

    runtime = TIDERuntime(model, router_path, config=TIDEConfig())

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = runtime.generate(inputs.input_ids, max_new_tokens=max_new_tokens, temperature=0)
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    return {
        "output": text,
        "stats": runtime.last_stats.summary() if runtime.last_stats else "N/A",
    }
