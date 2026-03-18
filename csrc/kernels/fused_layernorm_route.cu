/*
 * Fused RMSNorm + Router MLP + Threshold kernel.
 *
 * One block per token. Each block:
 *   1. RMSNorm reduction (sum of squares -> variance -> normalize)
 *   2. Down-projection (hidden_dim -> bottleneck_dim) fused with normalization
 *   3. SiLU activation
 *   4. Up-projection (bottleneck_dim -> 1) + sigmoid
 *
 * Grid:  (batch_size, 1, 1)
 * Block: (256, 1, 1)
 *
 * Input is templated on scalar_t (float, __half, __nv_bfloat16) for memory access.
 * All accumulation is in float32. Router weights are always float32.
 */

#include "dtype_utils.cuh"
#include <cmath>

namespace tide {

template <typename scalar_t, int HIDDEN_DIM, int BOTTLENECK_DIM, int BLOCK_SIZE>
__global__ void fused_layernorm_route_kernel(
    const scalar_t* __restrict__ input,      // [N, HIDDEN_DIM]
    const float* __restrict__ norm_weight,    // [HIDDEN_DIM] (always float)
    const float* __restrict__ down_weight,    // [HIDDEN_DIM, BOTTLENECK_DIM] (transposed)
    const float* __restrict__ up_weight,      // [BOTTLENECK_DIM, 1] (transposed)
    float* __restrict__ scores,               // [N]
    float eps,
    int n_tokens
) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens) return;

    int tid = threadIdx.x;
    const int elements_per_thread = HIDDEN_DIM / BLOCK_SIZE;

    extern __shared__ float smem[];
    float* s_reduce = smem;           // [8] for cross-warp reduction
    float* s_intermediate = smem + 8; // [BOTTLENECK_DIM] for intermediate activations

    const scalar_t* token_input = input + token_idx * HIDDEN_DIM;

    // Step 1: Compute RMS norm variance = mean(x^2)
    float local_sq_sum = 0.0f;
    float local_vals[32]; // Max elements per thread
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = tid + i * BLOCK_SIZE;
        float val = load_as_float(token_input, idx);
        local_vals[i] = val;
        local_sq_sum += val * val;
    }

    float variance = block_reduce_sum(local_sq_sum, s_reduce, tid);
    variance = rsqrtf(variance / HIDDEN_DIM + eps);

    // Step 2: Normalize + Down projection (tiled)
    const int TILE_SIZE = 32;
    const int n_tiles = (BOTTLENECK_DIM + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < n_tiles; tile++) {
        int col_start = tile * TILE_SIZE;
        int col_end = min(col_start + TILE_SIZE, BOTTLENECK_DIM);

        float partial[32] = {0};
        #pragma unroll
        for (int i = 0; i < elements_per_thread; i++) {
            int row = tid + i * BLOCK_SIZE;
            float normed_val = local_vals[i] * variance * norm_weight[row];
            for (int c = 0; c < (col_end - col_start); c++) {
                partial[c] += normed_val * down_weight[row * BOTTLENECK_DIM + col_start + c];
            }
        }

        for (int c = 0; c < (col_end - col_start); c++) {
            float val = block_reduce_sum(partial[c], s_reduce, tid);
            if (tid == 0) {
                s_intermediate[col_start + c] = val;
            }
            __syncthreads();
        }
    }
    __syncthreads();

    // Step 3: SiLU activation (first BOTTLENECK_DIM threads)
    if (tid < BOTTLENECK_DIM) {
        float x = s_intermediate[tid];
        s_intermediate[tid] = x / (1.0f + expf(-x)); // SiLU = x * sigmoid(x)
    }
    __syncthreads();

    // Step 4: Up projection (dot product to scalar) + sigmoid
    if (tid == 0) {
        float dot = 0.0f;
        for (int i = 0; i < BOTTLENECK_DIM; i++) {
            dot += s_intermediate[i] * up_weight[i];
        }
        scores[token_idx] = 1.0f / (1.0f + expf(-dot)); // sigmoid
    }
}

// Generic fallback for non-templated hidden dims
template <typename scalar_t>
__global__ void fused_layernorm_route_generic(
    const scalar_t* __restrict__ input,
    const float* __restrict__ norm_weight,
    const float* __restrict__ down_weight,
    const float* __restrict__ up_weight,
    float* __restrict__ scores,
    float eps,
    int n_tokens,
    int hidden_dim,
    int bottleneck_dim
) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    extern __shared__ float smem[];
    float* s_reduce = smem;
    float* s_intermediate = smem + 8;

    const scalar_t* token_input = input + token_idx * hidden_dim;

    // Step 1: RMSNorm
    float local_sq_sum = 0.0f;
    for (int idx = tid; idx < hidden_dim; idx += block_size) {
        float val = load_as_float(token_input, idx);
        local_sq_sum += val * val;
    }

    float variance = block_reduce_sum(local_sq_sum, s_reduce, tid);
    variance = rsqrtf(variance / hidden_dim + eps);

    // Step 2: Normalize + Down projection
    for (int col = 0; col < bottleneck_dim; col++) {
        float partial = 0.0f;
        for (int row = tid; row < hidden_dim; row += block_size) {
            float normed = load_as_float(token_input, row) * variance * norm_weight[row];
            partial += normed * down_weight[row * bottleneck_dim + col];
        }
        float val = block_reduce_sum(partial, s_reduce, tid);
        if (tid == 0) {
            s_intermediate[col] = val;
        }
        __syncthreads();
    }
    __syncthreads();

    // Step 3: SiLU
    if (tid < bottleneck_dim) {
        float x = s_intermediate[tid];
        s_intermediate[tid] = x / (1.0f + expf(-x));
    }
    __syncthreads();

    // Step 4: Up projection + sigmoid
    if (tid == 0) {
        float dot = 0.0f;
        for (int i = 0; i < bottleneck_dim; i++) {
            dot += s_intermediate[i] * up_weight[i];
        }
        scores[token_idx] = 1.0f / (1.0f + expf(-dot));
    }
}

} // namespace tide

// C++ entry point — templated on scalar_t, dispatched from torch_bindings.cpp
template <typename scalar_t>
void fused_layernorm_route_typed(
    const scalar_t* input,
    const float* norm_weight,
    const float* down_weight,
    const float* up_weight,
    float* scores,
    float eps,
    int n_tokens,
    int hidden_dim,
    int bottleneck_dim
) {
    const int block_size = 256;
    int smem_size = (8 + bottleneck_dim) * sizeof(float);

    // Template specializations for common hidden_dim / bottleneck_dim combos
    if (hidden_dim == 2048 && bottleneck_dim == 64) {
        tide::fused_layernorm_route_kernel<scalar_t, 2048, 64, 256>
            <<<n_tokens, block_size, smem_size>>>(
                input, norm_weight, down_weight, up_weight, scores, eps, n_tokens);
    } else if (hidden_dim == 2048 && bottleneck_dim == 128) {
        tide::fused_layernorm_route_kernel<scalar_t, 2048, 128, 256>
            <<<n_tokens, block_size, smem_size>>>(
                input, norm_weight, down_weight, up_weight, scores, eps, n_tokens);
    } else if (hidden_dim == 3072 && bottleneck_dim == 128) {
        tide::fused_layernorm_route_kernel<scalar_t, 3072, 128, 256>
            <<<n_tokens, block_size, smem_size>>>(
                input, norm_weight, down_weight, up_weight, scores, eps, n_tokens);
    } else if (hidden_dim == 4096 && bottleneck_dim == 128) {
        tide::fused_layernorm_route_kernel<scalar_t, 4096, 128, 256>
            <<<n_tokens, block_size, smem_size>>>(
                input, norm_weight, down_weight, up_weight, scores, eps, n_tokens);
    } else if (hidden_dim == 4096 && bottleneck_dim == 256) {
        tide::fused_layernorm_route_kernel<scalar_t, 4096, 256, 256>
            <<<n_tokens, block_size, smem_size>>>(
                input, norm_weight, down_weight, up_weight, scores, eps, n_tokens);
    } else if (hidden_dim == 5120 && bottleneck_dim == 128) {
        tide::fused_layernorm_route_kernel<scalar_t, 5120, 128, 256>
            <<<n_tokens, block_size, smem_size>>>(
                input, norm_weight, down_weight, up_weight, scores, eps, n_tokens);
    } else if (hidden_dim == 8192 && bottleneck_dim == 128) {
        tide::fused_layernorm_route_kernel<scalar_t, 8192, 128, 256>
            <<<n_tokens, block_size, smem_size>>>(
                input, norm_weight, down_weight, up_weight, scores, eps, n_tokens);
    } else if (hidden_dim == 8192 && bottleneck_dim == 256) {
        tide::fused_layernorm_route_kernel<scalar_t, 8192, 256, 256>
            <<<n_tokens, block_size, smem_size>>>(
                input, norm_weight, down_weight, up_weight, scores, eps, n_tokens);
    } else {
        tide::fused_layernorm_route_generic<scalar_t>
            <<<n_tokens, block_size, smem_size>>>(
                input, norm_weight, down_weight, up_weight, scores,
                eps, n_tokens, hidden_dim, bottleneck_dim);
    }
}

// Named entry points (called from torch_bindings.cpp)
void fused_layernorm_route_typed_float(
    const float* input, const float* norm_weight,
    const float* down_weight, const float* up_weight,
    float* scores, float eps, int n_tokens, int hidden_dim, int bottleneck_dim
) { fused_layernorm_route_typed<float>(input, norm_weight, down_weight, up_weight, scores, eps, n_tokens, hidden_dim, bottleneck_dim); }

void fused_layernorm_route_typed_half(
    const __half* input, const float* norm_weight,
    const float* down_weight, const float* up_weight,
    float* scores, float eps, int n_tokens, int hidden_dim, int bottleneck_dim
) { fused_layernorm_route_typed<__half>(input, norm_weight, down_weight, up_weight, scores, eps, n_tokens, hidden_dim, bottleneck_dim); }

void fused_layernorm_route_typed_bf16(
    const __nv_bfloat16* input, const float* norm_weight,
    const float* down_weight, const float* up_weight,
    float* scores, float eps, int n_tokens, int hidden_dim, int bottleneck_dim
) { fused_layernorm_route_typed<__nv_bfloat16>(input, norm_weight, down_weight, up_weight, scores, eps, n_tokens, hidden_dim, bottleneck_dim); }
