/*
 * Exit scatter kernel: copies exited token hidden states to their
 * original positions in the output buffer.
 *
 * Grid:  (n_exited,)
 * Block: (256,)
 * Each block handles one token.
 *
 * Templated on scalar_t for mixed-precision support.
 */

#include "dtype_utils.cuh"

namespace tide {

template <typename scalar_t>
__global__ void exit_scatter_kernel(
    const scalar_t* __restrict__ exited_states,   // [N_exit, D]
    const int* __restrict__ exit_positions,        // [N_exit]
    scalar_t* __restrict__ output_buffer,          // [N_total, D]
    int n_exited,
    int hidden_dim
) {
    int exit_idx = blockIdx.x;
    if (exit_idx >= n_exited) return;

    int tid = threadIdx.x;
    int out_pos = exit_positions[exit_idx];

    const scalar_t* src = exited_states + exit_idx * hidden_dim;
    scalar_t* dst = output_buffer + out_pos * hidden_dim;

    // Copy in native dtype for bandwidth savings
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

} // namespace tide

// C++ entry point — templated
template <typename scalar_t>
void exit_scatter_typed(
    const scalar_t* exited_states,
    const int* exit_positions,
    scalar_t* output_buffer,
    int n_exited,
    int hidden_dim
) {
    if (n_exited == 0) return;
    tide::exit_scatter_kernel<scalar_t><<<n_exited, 256>>>(
        exited_states, exit_positions, output_buffer, n_exited, hidden_dim);
}

// Named entry points
void exit_scatter_typed_float(
    const float* exited_states, const int* exit_positions,
    float* output_buffer, int n_exited, int hidden_dim
) { exit_scatter_typed<float>(exited_states, exit_positions, output_buffer, n_exited, hidden_dim); }

void exit_scatter_typed_half(
    const __half* exited_states, const int* exit_positions,
    __half* output_buffer, int n_exited, int hidden_dim
) { exit_scatter_typed<__half>(exited_states, exit_positions, output_buffer, n_exited, hidden_dim); }

void exit_scatter_typed_bf16(
    const __nv_bfloat16* exited_states, const int* exit_positions,
    __nv_bfloat16* output_buffer, int n_exited, int hidden_dim
) { exit_scatter_typed<__nv_bfloat16>(exited_states, exit_positions, output_buffer, n_exited, hidden_dim); }
