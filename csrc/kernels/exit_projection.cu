/*
 * Exit projection kernel.
 *
 * Two modes:
 *   A) Identity + RMSNorm: fused normalize + scatter
 *   B) Learned projection: RMSNorm -> GEMM -> scatter (uses cuBLAS externally)
 *
 * This file implements Mode A (fused RMSNorm + scatter).
 *
 * Templated on scalar_t for mixed-precision support.
 * Reads input in native dtype, accumulates in float32, writes output in native dtype.
 */

#include "dtype_utils.cuh"
#include <cmath>

namespace tide {

// Mode A: Fused RMSNorm + scatter to output positions
template <typename scalar_t>
__global__ void exit_projection_identity_kernel(
    const scalar_t* __restrict__ exited_states,   // [N_exit, D]
    const float* __restrict__ norm_weight,         // [D] (always float)
    const int* __restrict__ exit_positions,        // [N_exit]
    scalar_t* __restrict__ output_buffer,          // [N_total, D]
    float eps,
    int n_exited,
    int hidden_dim
) {
    int exit_idx = blockIdx.x;
    if (exit_idx >= n_exited) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    extern __shared__ float smem[];

    const scalar_t* src = exited_states + exit_idx * hidden_dim;
    int out_pos = exit_positions[exit_idx];
    scalar_t* dst = output_buffer + out_pos * hidden_dim;

    // Compute RMS norm (accumulate in float32)
    float local_sq_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += block_size) {
        float val = load_as_float(src, i);
        local_sq_sum += val * val;
    }

    float variance = block_reduce_sum(local_sq_sum, smem, tid);
    float inv_rms = rsqrtf(variance / hidden_dim + eps);

    // Normalize and scatter (write in native dtype)
    for (int i = tid; i < hidden_dim; i += block_size) {
        float val = load_as_float(src, i);
        store_from_float(dst, i, val * inv_rms * norm_weight[i]);
    }
}

// Mode B stub: RMSNorm only (GEMM done via cuBLAS externally)
template <typename scalar_t>
__global__ void exit_rmsnorm_kernel(
    const scalar_t* __restrict__ input,    // [N, D]
    const float* __restrict__ weight,      // [D] (always float)
    scalar_t* __restrict__ output,         // [N, D]
    float eps,
    int n_tokens,
    int hidden_dim
) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    extern __shared__ float smem[];

    const scalar_t* src = input + token_idx * hidden_dim;
    scalar_t* dst = output + token_idx * hidden_dim;

    float local_sq_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += block_size) {
        float val = load_as_float(src, i);
        local_sq_sum += val * val;
    }

    float variance = block_reduce_sum(local_sq_sum, smem, tid);
    float inv_rms = rsqrtf(variance / hidden_dim + eps);

    for (int i = tid; i < hidden_dim; i += block_size) {
        float val = load_as_float(src, i);
        store_from_float(dst, i, val * inv_rms * weight[i]);
    }
}

} // namespace tide

// C++ entry points — templated
template <typename scalar_t>
void exit_projection_identity_typed(
    const scalar_t* exited_states,
    const float* norm_weight,
    const int* exit_positions,
    scalar_t* output_buffer,
    float eps,
    int n_exited,
    int hidden_dim
) {
    if (n_exited == 0) return;
    int smem_size = 8 * sizeof(float);
    tide::exit_projection_identity_kernel<scalar_t><<<n_exited, 256, smem_size>>>(
        exited_states, norm_weight, exit_positions, output_buffer,
        eps, n_exited, hidden_dim);
}

template <typename scalar_t>
void exit_rmsnorm_typed(
    const scalar_t* input,
    const float* weight,
    scalar_t* output,
    float eps,
    int n_tokens,
    int hidden_dim
) {
    if (n_tokens == 0) return;
    int smem_size = 8 * sizeof(float);
    tide::exit_rmsnorm_kernel<scalar_t><<<n_tokens, 256, smem_size>>>(
        input, weight, output, eps, n_tokens, hidden_dim);
}

// Named entry points
void exit_projection_identity_typed_float(
    const float* exited_states, const float* norm_weight,
    const int* exit_positions, float* output_buffer,
    float eps, int n_exited, int hidden_dim
) { exit_projection_identity_typed<float>(exited_states, norm_weight, exit_positions, output_buffer, eps, n_exited, hidden_dim); }

void exit_projection_identity_typed_half(
    const __half* exited_states, const float* norm_weight,
    const int* exit_positions, __half* output_buffer,
    float eps, int n_exited, int hidden_dim
) { exit_projection_identity_typed<__half>(exited_states, norm_weight, exit_positions, output_buffer, eps, n_exited, hidden_dim); }

void exit_projection_identity_typed_bf16(
    const __nv_bfloat16* exited_states, const float* norm_weight,
    const int* exit_positions, __nv_bfloat16* output_buffer,
    float eps, int n_exited, int hidden_dim
) { exit_projection_identity_typed<__nv_bfloat16>(exited_states, norm_weight, exit_positions, output_buffer, eps, n_exited, hidden_dim); }

void exit_rmsnorm_typed_float(
    const float* input, const float* weight, float* output,
    float eps, int n_tokens, int hidden_dim
) { exit_rmsnorm_typed<float>(input, weight, output, eps, n_tokens, hidden_dim); }

void exit_rmsnorm_typed_half(
    const __half* input, const float* weight, __half* output,
    float eps, int n_tokens, int hidden_dim
) { exit_rmsnorm_typed<__half>(input, weight, output, eps, n_tokens, hidden_dim); }

void exit_rmsnorm_typed_bf16(
    const __nv_bfloat16* input, const float* weight, __nv_bfloat16* output,
    float eps, int n_tokens, int hidden_dim
) { exit_rmsnorm_typed<__nv_bfloat16>(input, weight, output, eps, n_tokens, hidden_dim); }
