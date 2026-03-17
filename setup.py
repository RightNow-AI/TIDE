import os
import subprocess
from setuptools import setup


def check_nvcc():
    """Check if CUDA toolkit is available."""
    try:
        subprocess.check_output(["nvcc", "--version"])
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def detect_cuda_arch_flags():
    """Detect CUDA architecture flags for compilation.

    Priority:
      1. TIDE_CUDA_ARCH env var (e.g. "8.6")
      2. TORCH_CUDA_ARCH_LIST env var (e.g. "8.0;9.0" or "8.0 9.0")
      3. Query torch.cuda.get_device_capability() if GPU available
      4. Fallback: compile for sm_70 through sm_90
    """
    # Check TIDE-specific env var first
    tide_arch = os.environ.get("TIDE_CUDA_ARCH")
    if tide_arch:
        arches = [a.strip() for a in tide_arch.replace(";", " ").split()]
        return _arches_to_gencode(arches)

    # Check PyTorch standard env var
    torch_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if torch_arch:
        arches = [a.strip() for a in torch_arch.replace(";", " ").split()]
        return _arches_to_gencode(arches)

    # Try to detect from available GPU
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            arch = f"{cap[0]}.{cap[1]}"
            return _arches_to_gencode([arch])
    except (ImportError, RuntimeError):
        pass

    # Fallback: broad range from V100 through H100
    return _arches_to_gencode(["7.0", "7.5", "8.0", "8.6", "8.9", "9.0"])


def _arches_to_gencode(arches):
    """Convert architecture strings like '8.6' to -gencode flags."""
    flags = []
    for arch in arches:
        # Handle formats: "8.6", "86", "8.6+PTX"
        ptx = "+PTX" in arch
        arch_clean = arch.replace("+PTX", "").strip()
        if "." in arch_clean:
            major, minor = arch_clean.split(".")
            compute = f"{major}{minor}"
        else:
            compute = arch_clean

        flags.append(f"-gencode=arch=compute_{compute},code=sm_{compute}")
        if ptx:
            flags.append(f"-gencode=arch=compute_{compute},code=compute_{compute}")
    return flags


ext_modules = []
cmdclass = {}

if check_nvcc() and os.environ.get("TIDE_NO_CUDA", "0") != "1":
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension

    cuda_arch_flags = detect_cuda_arch_flags()

    ext_modules = [
        CUDAExtension(
            name="TIDE._C",
            sources=[
                "csrc/extensions/torch_bindings.cpp",
                "csrc/kernels/fused_layernorm_route.cu",
                "csrc/kernels/batch_compact.cu",
                "csrc/kernels/exit_scatter.cu",
                "csrc/kernels/exit_projection.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    *cuda_arch_flags,
                    "-lineinfo",
                    "--threads=4",
                ],
            },
        ),
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
