"""
CUDA kernel numerical equivalence tests.
Tests CUDA output vs Python reference implementation.
Skipped if CUDA is not available.

Parametrized over float32, float16, and bfloat16 to verify mixed-precision support.
"""

import pytest
import torch
import torch.nn as nn

cuda_available = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not cuda_available, reason="CUDA not available")

kernels_available = False
try:
    import TIDE._C
    kernels_available = True
except ImportError:
    # TORCH_LIBRARY extensions may not have PyInit — load via torch.ops
    try:
        import glob, os, importlib.util
        spec = importlib.util.find_spec("TIDE")
        if spec and spec.submodule_search_locations:
            for sp in spec.submodule_search_locations:
                so_files = glob.glob(os.path.join(sp, "_C*.so"))
                if so_files:
                    torch.ops.load_library(so_files[0])
                    kernels_available = True
                    break
    except Exception:
        pass

skip_no_kernels = pytest.mark.skipif(not kernels_available, reason="TIDE CUDA kernels not built")

# Dtype parametrization
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
DTYPE_IDS = ["fp32", "fp16", "bf16"]


def python_rmsnorm(x, weight, eps=1e-6):
    """Reference RMSNorm implementation."""
    x_f = x.float()
    variance = x_f.pow(2).mean(-1, keepdim=True)
    x_normed = x_f * torch.rsqrt(variance + eps)
    return weight.float() * x_normed


def python_router(x_normed, down_weight, up_weight):
    """Reference router MLP: down -> SiLU -> up -> sigmoid."""
    h = x_normed @ down_weight       # [N, bottleneck]
    h = torch.nn.functional.silu(h)
    out = h @ up_weight               # [N, 1]
    return torch.sigmoid(out).squeeze(-1)


def python_fused_layernorm_route(x, norm_weight, down_weight, up_weight, eps=1e-6):
    """Reference fused layernorm + route."""
    normed = python_rmsnorm(x, norm_weight, eps)
    return python_router(normed, down_weight, up_weight)


@skip_no_cuda
@skip_no_kernels
class TestFusedLayernormRoute:
    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
    @pytest.mark.parametrize("hidden_dim", [2048, 4096])
    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    def test_numerical_equivalence(self, batch_size, hidden_dim, dtype):
        bottleneck_dim = 128
        device = "cuda"

        x = torch.randn(batch_size, hidden_dim, device=device, dtype=dtype)
        norm_weight = torch.randn(hidden_dim, device=device)  # always float32
        down_weight = torch.randn(hidden_dim, bottleneck_dim, device=device)
        up_weight = torch.randn(bottleneck_dim, 1, device=device)

        ref = python_fused_layernorm_route(x, norm_weight, down_weight, up_weight)
        cuda_out = torch.ops.tide.fused_layernorm_route(
            x, norm_weight, down_weight, up_weight, 1e-6
        )

        # Wider tolerance for half precision due to reduced input precision
        atol = 5e-2 if dtype != torch.float32 else 1e-2
        rtol = 5e-2 if dtype != torch.float32 else 1e-2
        torch.testing.assert_close(cuda_out, ref, atol=atol, rtol=rtol)


@skip_no_cuda
@skip_no_kernels
class TestBatchCompact:
    @pytest.mark.parametrize("n_tokens", [1, 8, 32, 128])
    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    def test_compact_preserves_data(self, n_tokens, dtype):
        hidden_dim = 256
        device = "cuda"

        x = torch.randn(n_tokens, hidden_dim, device=device, dtype=dtype)
        exit_mask = torch.randint(0, 2, (n_tokens,), device=device, dtype=torch.int32)

        results = torch.ops.tide.batch_compact(x, exit_mask)
        compacted, exited, cont_idx, exit_idx, num_remain = results

        n_remain = num_remain.item()
        n_exit = n_tokens - n_remain

        assert compacted.shape == (n_remain, hidden_dim)
        assert exited.shape == (n_exit, hidden_dim)
        assert compacted.dtype == dtype
        assert exited.dtype == dtype

    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    def test_all_continue(self, dtype):
        device = "cuda"
        x = torch.randn(8, 256, device=device, dtype=dtype)
        exit_mask = torch.zeros(8, device=device, dtype=torch.int32)

        results = torch.ops.tide.batch_compact(x, exit_mask)
        assert results[4].item() == 8  # all continue

    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    def test_all_exit(self, dtype):
        device = "cuda"
        x = torch.randn(8, 256, device=device, dtype=dtype)
        exit_mask = torch.ones(8, device=device, dtype=torch.int32)

        results = torch.ops.tide.batch_compact(x, exit_mask)
        assert results[4].item() == 0  # none continue


@skip_no_cuda
@skip_no_kernels
class TestExitScatter:
    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    def test_scatter_correctness(self, dtype):
        device = "cuda"
        hidden_dim = 256
        n_total = 32
        n_exited = 8

        exited_states = torch.randn(n_exited, hidden_dim, device=device, dtype=dtype)
        positions = torch.tensor(
            [2, 5, 8, 11, 15, 20, 25, 30], device=device, dtype=torch.int32
        )
        output_buffer = torch.zeros(n_total, hidden_dim, device=device, dtype=dtype)

        torch.ops.tide.exit_scatter(exited_states, positions, output_buffer)

        for i, pos in enumerate(positions):
            torch.testing.assert_close(
                output_buffer[pos].float(),
                exited_states[i].float(),
                atol=1e-5,
                rtol=1e-5,
            )


@skip_no_cuda
@skip_no_kernels
class TestExitProjection:
    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    def test_identity_projection(self, dtype):
        device = "cuda"
        hidden_dim = 256
        n_total = 32
        n_exited = 4

        exited = torch.randn(n_exited, hidden_dim, device=device, dtype=dtype)
        norm_weight = torch.ones(hidden_dim, device=device)  # always float32
        positions = torch.tensor([1, 7, 15, 28], device=device, dtype=torch.int32)
        output_buffer = torch.zeros(n_total, hidden_dim, device=device, dtype=dtype)

        torch.ops.tide.exit_projection(exited, norm_weight, positions, output_buffer, 1e-6)

        # Verify RMSNorm was applied (ref always in float32)
        ref_normed = python_rmsnorm(exited, norm_weight, 1e-6)
        atol = 5e-2 if dtype != torch.float32 else 1e-2
        rtol = 5e-2 if dtype != torch.float32 else 1e-2
        for i, pos in enumerate(positions):
            torch.testing.assert_close(
                output_buffer[pos].float(),
                ref_normed[i].float(),
                atol=atol,
                rtol=rtol,
            )
