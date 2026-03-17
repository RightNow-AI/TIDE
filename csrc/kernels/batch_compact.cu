/*
 * Batch compaction kernel for TIDE.
 *
 * Two-path design:
 *   - Small batch (<=32): Single warp using __ballot_sync + __popc
 *   - Large batch (>32): Two-kernel approach with prefix sum + scatter
 *
 * Separates active (continuing) and exited tokens based on an exit mask.
 *
 * Templated on scalar_t for mixed-precision memory access.
 * Copy operations work in the native dtype for memory bandwidth savings.
 */

#include "dtype_utils.cuh"
#include <cstdint>

namespace tide {

// Small batch: single warp compaction using ballot
template <typename scalar_t>
__global__ void batch_compact_small(
    const scalar_t* __restrict__ input,     // [N, D]
    const int* __restrict__ exit_mask,       // [N] 0=continue, 1=exit
    scalar_t* __restrict__ compacted,        // [N_continue, D]
    scalar_t* __restrict__ exited,           // [N_exit, D]
    int* __restrict__ continue_indices,      // [N_continue]
    int* __restrict__ exit_indices,          // [N_exit]
    int* __restrict__ num_remaining,         // [1]
    int n_tokens,
    int hidden_dim
) {
    int tid = threadIdx.x;
    if (tid >= n_tokens) return;

    int is_exit = exit_mask[tid];

    // Ballot: get bitmask of exiting tokens across warp
    unsigned int exit_ballot = __ballot_sync(0xFFFFFFFF, is_exit);
    unsigned int continue_ballot = ~exit_ballot & ((1u << n_tokens) - 1);

    // Count exits/continues before this thread
    unsigned int mask_before = (1u << tid) - 1;
    int exit_offset = __popc(exit_ballot & mask_before);
    int continue_offset = __popc(continue_ballot & mask_before);

    // Copy hidden state to appropriate output
    if (is_exit) {
        exit_indices[exit_offset] = tid;
        for (int d = 0; d < hidden_dim; d++) {
            exited[exit_offset * hidden_dim + d] = input[tid * hidden_dim + d];
        }
    } else {
        continue_indices[continue_offset] = tid;
        for (int d = 0; d < hidden_dim; d++) {
            compacted[continue_offset * hidden_dim + d] = input[tid * hidden_dim + d];
        }
    }

    // Thread 0 writes count
    if (tid == 0) {
        *num_remaining = __popc(continue_ballot);
    }
}

// Large batch kernel A: compute prefix sums
__global__ void compute_prefix_sums(
    const int* __restrict__ exit_mask,   // [N]
    int* __restrict__ continue_prefix,    // [N]
    int* __restrict__ exit_prefix,        // [N]
    int* __restrict__ num_remaining,      // [1]
    int n_tokens
) {
    extern __shared__ int smem[];
    int* s_continue = smem;          // [blockDim.x]
    int* s_exit = smem + blockDim.x; // [blockDim.x]

    int tid = threadIdx.x;
    if (tid >= n_tokens) return;

    int is_exit = exit_mask[tid];
    s_continue[tid] = is_exit ? 0 : 1;
    s_exit[tid] = is_exit ? 1 : 0;
    __syncthreads();

    // Inclusive prefix sum (Hillis-Steele)
    for (int stride = 1; stride < n_tokens; stride <<= 1) {
        int c_val = (tid >= stride) ? s_continue[tid - stride] : 0;
        int e_val = (tid >= stride) ? s_exit[tid - stride] : 0;
        __syncthreads();
        s_continue[tid] += c_val;
        s_exit[tid] += e_val;
        __syncthreads();
    }

    // Convert to exclusive prefix sum
    continue_prefix[tid] = s_continue[tid] - (is_exit ? 0 : 1);
    exit_prefix[tid] = s_exit[tid] - (is_exit ? 1 : 0);

    if (tid == n_tokens - 1) {
        *num_remaining = s_continue[tid];
    }
}

// Large batch kernel B: scatter using prefix sums
template <typename scalar_t>
__global__ void scatter_compact(
    const scalar_t* __restrict__ input,
    const int* __restrict__ exit_mask,
    const int* __restrict__ continue_prefix,
    const int* __restrict__ exit_prefix,
    scalar_t* __restrict__ compacted,
    scalar_t* __restrict__ exited,
    int* __restrict__ continue_indices,
    int* __restrict__ exit_indices,
    int n_tokens,
    int hidden_dim
) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens) return;

    int tid = threadIdx.x;
    int is_exit = exit_mask[token_idx];

    scalar_t* dst;
    int dst_row;
    int* idx_dst;

    if (is_exit) {
        dst_row = exit_prefix[token_idx];
        dst = exited + dst_row * hidden_dim;
        idx_dst = exit_indices;
        if (tid == 0) idx_dst[dst_row] = token_idx;
    } else {
        dst_row = continue_prefix[token_idx];
        dst = compacted + dst_row * hidden_dim;
        idx_dst = continue_indices;
        if (tid == 0) idx_dst[dst_row] = token_idx;
    }

    const scalar_t* src = input + token_idx * hidden_dim;

    // Vectorized copy — use native dtype for bandwidth savings
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

} // namespace tide

// C++ entry points — templated on scalar_t
template <typename scalar_t>
void batch_compact_typed(
    const scalar_t* input,
    const int* exit_mask,
    scalar_t* compacted,
    scalar_t* exited,
    int* continue_indices,
    int* exit_indices,
    int* num_remaining,
    int n_tokens,
    int hidden_dim
) {
    if (n_tokens <= 32) {
        tide::batch_compact_small<scalar_t><<<1, 32>>>(
            input, exit_mask, compacted, exited,
            continue_indices, exit_indices, num_remaining,
            n_tokens, hidden_dim);
    } else {
        // Allocate prefix sum buffers
        int *continue_prefix, *exit_prefix;
        cudaMalloc(&continue_prefix, n_tokens * sizeof(int));
        cudaMalloc(&exit_prefix, n_tokens * sizeof(int));

        int smem_size = 2 * n_tokens * sizeof(int);
        tide::compute_prefix_sums<<<1, n_tokens, smem_size>>>(
            exit_mask, continue_prefix, exit_prefix, num_remaining, n_tokens);

        tide::scatter_compact<scalar_t><<<n_tokens, 256>>>(
            input, exit_mask, continue_prefix, exit_prefix,
            compacted, exited, continue_indices, exit_indices,
            n_tokens, hidden_dim);

        cudaFree(continue_prefix);
        cudaFree(exit_prefix);
    }
}

// Named entry points
void batch_compact_typed_float(
    const float* input, const int* exit_mask, float* compacted, float* exited,
    int* continue_indices, int* exit_indices, int* num_remaining, int n_tokens, int hidden_dim
) { batch_compact_typed<float>(input, exit_mask, compacted, exited, continue_indices, exit_indices, num_remaining, n_tokens, hidden_dim); }

void batch_compact_typed_half(
    const __half* input, const int* exit_mask, __half* compacted, __half* exited,
    int* continue_indices, int* exit_indices, int* num_remaining, int n_tokens, int hidden_dim
) { batch_compact_typed<__half>(input, exit_mask, compacted, exited, continue_indices, exit_indices, num_remaining, n_tokens, hidden_dim); }

void batch_compact_typed_bf16(
    const __nv_bfloat16* input, const int* exit_mask, __nv_bfloat16* compacted, __nv_bfloat16* exited,
    int* continue_indices, int* exit_indices, int* num_remaining, int n_tokens, int hidden_dim
) { batch_compact_typed<__nv_bfloat16>(input, exit_mask, compacted, exited, continue_indices, exit_indices, num_remaining, n_tokens, hidden_dim); }
