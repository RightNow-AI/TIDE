/*
 * PyTorch C++ extension bindings for TIDE CUDA kernels.
 * Uses TORCH_LIBRARY registration for torch.compile compatibility.
 *
 * Supports float32, float16, and bfloat16 inputs via AT_DISPATCH.
 * Router weights and norm weights are always float32.
 * Scores/outputs are always float32.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Forward declarations of CUDA entry points (use CUDA native types)
void fused_layernorm_route_typed_float(
    const float* input, const float* norm_weight,
    const float* down_weight, const float* up_weight,
    float* scores, float eps,
    int n_tokens, int hidden_dim, int bottleneck_dim);
void fused_layernorm_route_typed_half(
    const __half* input, const float* norm_weight,
    const float* down_weight, const float* up_weight,
    float* scores, float eps,
    int n_tokens, int hidden_dim, int bottleneck_dim);
void fused_layernorm_route_typed_bf16(
    const __nv_bfloat16* input, const float* norm_weight,
    const float* down_weight, const float* up_weight,
    float* scores, float eps,
    int n_tokens, int hidden_dim, int bottleneck_dim);

void batch_compact_typed_float(
    const float* input, const int* exit_mask,
    float* compacted, float* exited,
    int* continue_indices, int* exit_indices, int* num_remaining,
    int n_tokens, int hidden_dim);
void batch_compact_typed_half(
    const __half* input, const int* exit_mask,
    __half* compacted, __half* exited,
    int* continue_indices, int* exit_indices, int* num_remaining,
    int n_tokens, int hidden_dim);
void batch_compact_typed_bf16(
    const __nv_bfloat16* input, const int* exit_mask,
    __nv_bfloat16* compacted, __nv_bfloat16* exited,
    int* continue_indices, int* exit_indices, int* num_remaining,
    int n_tokens, int hidden_dim);

void exit_scatter_typed_float(
    const float* exited_states, const int* exit_positions,
    float* output_buffer, int n_exited, int hidden_dim);
void exit_scatter_typed_half(
    const __half* exited_states, const int* exit_positions,
    __half* output_buffer, int n_exited, int hidden_dim);
void exit_scatter_typed_bf16(
    const __nv_bfloat16* exited_states, const int* exit_positions,
    __nv_bfloat16* output_buffer, int n_exited, int hidden_dim);

void exit_projection_identity_typed_float(
    const float* exited_states, const float* norm_weight,
    const int* exit_positions, float* output_buffer,
    float eps, int n_exited, int hidden_dim);
void exit_projection_identity_typed_half(
    const __half* exited_states, const float* norm_weight,
    const int* exit_positions, __half* output_buffer,
    float eps, int n_exited, int hidden_dim);
void exit_projection_identity_typed_bf16(
    const __nv_bfloat16* exited_states, const float* norm_weight,
    const int* exit_positions, __nv_bfloat16* output_buffer,
    float eps, int n_exited, int hidden_dim);

// --- Wrapper functions ---

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor fused_layernorm_route(
    torch::Tensor input,        // [N, D] — float32/fp16/bf16
    torch::Tensor norm_weight,  // [D] — always float32
    torch::Tensor down_weight,  // [D, B] (pre-transposed) — always float32
    torch::Tensor up_weight,    // [B, 1] (pre-transposed) — always float32
    double eps
) {
    CHECK_INPUT(input);
    CHECK_INPUT(norm_weight);
    CHECK_INPUT(down_weight);
    CHECK_INPUT(up_weight);

    int n_tokens = input.size(0);
    int hidden_dim = input.size(1);
    int bottleneck_dim = down_weight.size(1);

    TORCH_CHECK(input.device() == norm_weight.device(), "Device mismatch");
    TORCH_CHECK(norm_weight.size(0) == hidden_dim, "norm_weight dim mismatch");

    at::cuda::CUDAGuard device_guard(input.device());

    // Scores are always float32
    auto scores = torch::empty({n_tokens}, input.options().dtype(torch::kFloat32));
    // Weights are always float32
    auto norm_f = norm_weight.to(torch::kFloat32).contiguous();
    auto down_f = down_weight.to(torch::kFloat32).contiguous();
    auto up_f = up_weight.to(torch::kFloat32).contiguous();

    auto input_c = input.contiguous();

    switch (input.scalar_type()) {
        case at::ScalarType::Float:
            fused_layernorm_route_typed_float(
                input_c.data_ptr<float>(), norm_f.data_ptr<float>(),
                down_f.data_ptr<float>(), up_f.data_ptr<float>(),
                scores.data_ptr<float>(), static_cast<float>(eps),
                n_tokens, hidden_dim, bottleneck_dim);
            break;
        case at::ScalarType::Half:
            fused_layernorm_route_typed_half(
                reinterpret_cast<const __half*>(input_c.data_ptr<at::Half>()),
                norm_f.data_ptr<float>(), down_f.data_ptr<float>(),
                up_f.data_ptr<float>(), scores.data_ptr<float>(),
                static_cast<float>(eps), n_tokens, hidden_dim, bottleneck_dim);
            break;
        case at::ScalarType::BFloat16:
            fused_layernorm_route_typed_bf16(
                reinterpret_cast<const __nv_bfloat16*>(input_c.data_ptr<at::BFloat16>()),
                norm_f.data_ptr<float>(), down_f.data_ptr<float>(),
                up_f.data_ptr<float>(), scores.data_ptr<float>(),
                static_cast<float>(eps), n_tokens, hidden_dim, bottleneck_dim);
            break;
        default:
            TORCH_CHECK(false, "fused_layernorm_route: unsupported dtype ", input.scalar_type());
    }

    return scores;
}

std::vector<torch::Tensor> batch_compact(
    torch::Tensor input,     // [N, D] — float32/fp16/bf16
    torch::Tensor exit_mask  // [N] int32
) {
    CHECK_INPUT(input);
    CHECK_INPUT(exit_mask);

    int n_tokens = input.size(0);
    int hidden_dim = input.size(1);

    at::cuda::CUDAGuard device_guard(input.device());

    auto opts_i = input.options().dtype(torch::kInt32);

    // Output buffers match input dtype
    auto compacted = torch::empty({n_tokens, hidden_dim}, input.options());
    auto exited = torch::empty({n_tokens, hidden_dim}, input.options());
    auto continue_indices = torch::empty({n_tokens}, opts_i);
    auto exit_indices = torch::empty({n_tokens}, opts_i);
    auto num_remaining = torch::zeros({1}, opts_i);

    auto input_c = input.contiguous();
    auto mask_i = exit_mask.to(torch::kInt32).contiguous();

    switch (input.scalar_type()) {
        case at::ScalarType::Float:
            batch_compact_typed_float(
                input_c.data_ptr<float>(), mask_i.data_ptr<int>(),
                compacted.data_ptr<float>(), exited.data_ptr<float>(),
                continue_indices.data_ptr<int>(), exit_indices.data_ptr<int>(),
                num_remaining.data_ptr<int>(), n_tokens, hidden_dim);
            break;
        case at::ScalarType::Half:
            batch_compact_typed_half(
                reinterpret_cast<const __half*>(input_c.data_ptr<at::Half>()),
                mask_i.data_ptr<int>(),
                reinterpret_cast<__half*>(compacted.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(exited.data_ptr<at::Half>()),
                continue_indices.data_ptr<int>(), exit_indices.data_ptr<int>(),
                num_remaining.data_ptr<int>(), n_tokens, hidden_dim);
            break;
        case at::ScalarType::BFloat16:
            batch_compact_typed_bf16(
                reinterpret_cast<const __nv_bfloat16*>(input_c.data_ptr<at::BFloat16>()),
                mask_i.data_ptr<int>(),
                reinterpret_cast<__nv_bfloat16*>(compacted.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(exited.data_ptr<at::BFloat16>()),
                continue_indices.data_ptr<int>(), exit_indices.data_ptr<int>(),
                num_remaining.data_ptr<int>(), n_tokens, hidden_dim);
            break;
        default:
            TORCH_CHECK(false, "batch_compact: unsupported dtype ", input.scalar_type());
    }

    int n_remain = num_remaining.item<int>();
    return {
        compacted.slice(0, 0, n_remain),
        exited.slice(0, 0, n_tokens - n_remain),
        continue_indices.slice(0, 0, n_remain),
        exit_indices.slice(0, 0, n_tokens - n_remain),
        num_remaining,
    };
}

void exit_scatter(
    torch::Tensor exited_states,  // [N_exit, D] — float32/fp16/bf16
    torch::Tensor exit_positions, // [N_exit]
    torch::Tensor output_buffer   // [N_total, D] — same dtype as exited_states
) {
    CHECK_INPUT(exited_states);
    CHECK_INPUT(exit_positions);
    CHECK_INPUT(output_buffer);

    TORCH_CHECK(exited_states.scalar_type() == output_buffer.scalar_type(),
                "exited_states and output_buffer must have the same dtype");

    int n_exited = exited_states.size(0);
    int hidden_dim = exited_states.size(1);

    at::cuda::CUDAGuard device_guard(exited_states.device());

    auto states_c = exited_states.contiguous();
    auto pos_i = exit_positions.to(torch::kInt32).contiguous();

    switch (exited_states.scalar_type()) {
        case at::ScalarType::Float:
            exit_scatter_typed_float(
                states_c.data_ptr<float>(), pos_i.data_ptr<int>(),
                output_buffer.data_ptr<float>(), n_exited, hidden_dim);
            break;
        case at::ScalarType::Half:
            exit_scatter_typed_half(
                reinterpret_cast<const __half*>(states_c.data_ptr<at::Half>()),
                pos_i.data_ptr<int>(),
                reinterpret_cast<__half*>(output_buffer.data_ptr<at::Half>()),
                n_exited, hidden_dim);
            break;
        case at::ScalarType::BFloat16:
            exit_scatter_typed_bf16(
                reinterpret_cast<const __nv_bfloat16*>(states_c.data_ptr<at::BFloat16>()),
                pos_i.data_ptr<int>(),
                reinterpret_cast<__nv_bfloat16*>(output_buffer.data_ptr<at::BFloat16>()),
                n_exited, hidden_dim);
            break;
        default:
            TORCH_CHECK(false, "exit_scatter: unsupported dtype ", exited_states.scalar_type());
    }
}

void exit_projection(
    torch::Tensor exited_states,  // [N_exit, D] — float32/fp16/bf16
    torch::Tensor norm_weight,    // [D] — always float32
    torch::Tensor exit_positions, // [N_exit]
    torch::Tensor output_buffer,  // [N_total, D] — same dtype as exited_states
    double eps
) {
    CHECK_INPUT(exited_states);
    CHECK_INPUT(norm_weight);
    CHECK_INPUT(exit_positions);
    CHECK_INPUT(output_buffer);

    TORCH_CHECK(exited_states.scalar_type() == output_buffer.scalar_type(),
                "exited_states and output_buffer must have the same dtype");

    int n_exited = exited_states.size(0);
    int hidden_dim = exited_states.size(1);

    at::cuda::CUDAGuard device_guard(exited_states.device());

    auto states_c = exited_states.contiguous();
    auto norm_f = norm_weight.to(torch::kFloat32).contiguous();
    auto pos_i = exit_positions.to(torch::kInt32).contiguous();

    switch (exited_states.scalar_type()) {
        case at::ScalarType::Float:
            exit_projection_identity_typed_float(
                states_c.data_ptr<float>(), norm_f.data_ptr<float>(),
                pos_i.data_ptr<int>(), output_buffer.data_ptr<float>(),
                static_cast<float>(eps), n_exited, hidden_dim);
            break;
        case at::ScalarType::Half:
            exit_projection_identity_typed_half(
                reinterpret_cast<const __half*>(states_c.data_ptr<at::Half>()),
                norm_f.data_ptr<float>(), pos_i.data_ptr<int>(),
                reinterpret_cast<__half*>(output_buffer.data_ptr<at::Half>()),
                static_cast<float>(eps), n_exited, hidden_dim);
            break;
        case at::ScalarType::BFloat16:
            exit_projection_identity_typed_bf16(
                reinterpret_cast<const __nv_bfloat16*>(states_c.data_ptr<at::BFloat16>()),
                norm_f.data_ptr<float>(), pos_i.data_ptr<int>(),
                reinterpret_cast<__nv_bfloat16*>(output_buffer.data_ptr<at::BFloat16>()),
                static_cast<float>(eps), n_exited, hidden_dim);
            break;
        default:
            TORCH_CHECK(false, "exit_projection: unsupported dtype ", exited_states.scalar_type());
    }
}

// --- TORCH_LIBRARY registration ---

TORCH_LIBRARY(tide, m) {
    m.def("fused_layernorm_route(Tensor input, Tensor norm_weight, "
          "Tensor down_weight, Tensor up_weight, float eps) -> Tensor");
    m.def("batch_compact(Tensor input, Tensor exit_mask) -> Tensor[]");
    m.def("exit_scatter(Tensor exited_states, Tensor exit_positions, "
          "Tensor output_buffer) -> ()");
    m.def("exit_projection(Tensor exited_states, Tensor norm_weight, "
          "Tensor exit_positions, Tensor output_buffer, float eps) -> ()");
}

TORCH_LIBRARY_IMPL(tide, CUDA, m) {
    m.impl("fused_layernorm_route", &fused_layernorm_route);
    m.impl("batch_compact", &batch_compact);
    m.impl("exit_scatter", &exit_scatter);
    m.impl("exit_projection", &exit_projection);
}
