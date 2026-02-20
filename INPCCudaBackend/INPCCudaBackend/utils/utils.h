#pragma once

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

#define FLT_EPS 1e-8f

#define CHECK_CUDA(debug, name) \
if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << name << " " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

template <typename T>
__host__ __device__ T div_round_up(T value, T divisor) {
	return (value + divisor - 1) / divisor;
}

__device__ __forceinline__ const float3 transform_point_affine(
        float4 const *const matrix,
        const float3& position) {
    return make_float3(
        matrix[0].x * position.x + matrix[0].y * position.y + matrix[0].z * position.z + matrix[0].w,
        matrix[1].x * position.x + matrix[1].y * position.y + matrix[1].z * position.z + matrix[1].w,
        matrix[2].x * position.x + matrix[2].y * position.y + matrix[2].z * position.z + matrix[2].w
    );
}

template <typename T>
static void obtain(char*& blob, T*& ptr, std::size_t count, std::size_t alignment) {
    std::size_t offset = reinterpret_cast<std::uintptr_t>(blob) + alignment - 1 & ~(alignment - 1);
    ptr = reinterpret_cast<T*>(offset);
    blob = reinterpret_cast<char*>(ptr + count);
}

template<typename T>
size_t required(size_t N) {
    char* size = nullptr;
    T::from_blob(size, N);
    return size_t(size) + 128;
}
