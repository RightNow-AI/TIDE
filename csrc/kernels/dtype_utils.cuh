/*
 * Shared mixed-precision utilities for TIDE CUDA kernels.
 *
 * Provides load_as_float / store_from_float for transparent fp16/bf16 I/O
 * while keeping all accumulation in float32 for numerical stability.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace tide {

// ---------- Load/store helpers ----------

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* ptr, int idx);

template <>
__device__ __forceinline__ float load_as_float<float>(const float* ptr, int idx) {
    return ptr[idx];
}

template <>
__device__ __forceinline__ float load_as_float<__half>(const __half* ptr, int idx) {
    return __half2float(ptr[idx]);
}

template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(const __nv_bfloat16* ptr, int idx) {
    return __bfloat162float(ptr[idx]);
}

template <typename scalar_t>
__device__ __forceinline__ void store_from_float(scalar_t* ptr, int idx, float val);

template <>
__device__ __forceinline__ void store_from_float<float>(float* ptr, int idx, float val) {
    ptr[idx] = val;
}

template <>
__device__ __forceinline__ void store_from_float<__half>(__half* ptr, int idx, float val) {
    ptr[idx] = __float2half(val);
}

template <>
__device__ __forceinline__ void store_from_float<__nv_bfloat16>(__nv_bfloat16* ptr, int idx, float val) {
    ptr[idx] = __float2bfloat16(val);
}

// ---------- Warp/block reductions ----------

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared, int tid) {
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    val = warp_reduce_sum(val);

    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across warps
    val = (tid < 8) ? shared[tid] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }

    // Broadcast final result to all threads
    if (tid == 0) {
        shared[0] = val;
    }
    __syncthreads();

    return shared[0];
}

} // namespace tide
