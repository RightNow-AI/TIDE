"""Modal container image definitions for TIDE."""

import modal

# Core image: CUDA + torch + transformers (no flash-attn/vllm)
# Used for CI, testing, calibration
tide_core_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "wget", "ninja-build", "build-essential", "clang")
    .pip_install(
        "torch>=2.5.0",
        "numpy",
        "setuptools>=68.0",
        "wheel",
        "ninja",
    )
    .pip_install(
        "transformers>=4.48.0",
        "datasets",
        "pyyaml",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "triton>=3.1.0",
        "pytest",
        "pytest-cov",
    )
)

# Full image: adds flash-attn + vllm (for benchmarks)
tide_bench_image = (
    tide_core_image
    .pip_install(
        "flash-attn>=2.7.0",
        gpu="A10G",
    )
    .pip_install(
        "vllm>=0.7.0",
    )
)


def build_tide_image(
    tide_repo_path: str = "/root/TIDE",
    include_bench_deps: bool = False,
) -> modal.Image:
    """Build image with TIDE CUDA extensions compiled."""
    base = tide_bench_image if include_bench_deps else tide_core_image
    return (
        base
        .add_local_dir(".", remote_path=tide_repo_path, copy=True)
        .run_commands(
            f"cd {tide_repo_path} && TORCH_CUDA_ARCH_LIST='8.0;8.6;9.0' pip install --no-cache-dir '.[test]'",
            gpu="A10G",
        )
        .add_local_python_source("modal_setup")
    )
