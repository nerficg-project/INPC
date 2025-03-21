#pragma once

#include <cuda_fp16.h>

// half4

struct __align__(8) half4 { __half2 xy, zw; };

inline __device__ half4 make_half4(const __half2& xy, const __half2& zw) {
    half4 result;
    result.xy = xy;
    result.zw = zw;
    return result;
}

inline __device__ half4 make_half4(const __half& x, const __half& y, const __half& z, const __half& w) {
    half4 result;
    result.xy = __halves2half2(x, y);
    result.zw = __halves2half2(z, w);
    return result;
}

inline __device__ float4 half42float4(const half4& input) {
    const auto [x, y] = __half22float2(input.xy);
    const auto [z, w] = __half22float2(input.zw);
    return make_float4(x, y, z, w);
}

inline __device__ half4 float42half4(const float4& input) {
    const __half2 xy = __float22half2_rn(make_float2(input.x, input.y));
    const __half2 zw = __float22half2_rn(make_float2(input.z, input.w));
    return make_half4(xy, zw);
}

// half3

struct __align__(2) half3 { __half x, y, z; };

inline __device__ half3 make_half3(const __half& x, const __half& y, const __half& z) {
    half3 result;
    result.x = x;
    result.y = y;
    result.z = z;
    return result;
}
inline __device__ float3 half32float3(const half3& input) {
    const auto [x, y] = __half22float2(__halves2half2(input.x, input.y));
    const float z = __half2float(input.z);
    return make_float3(x, y, z);
}
inline __device__ half3 float32half3(const float3& input) {
    const __half2 xy = __float22half2_rn(make_float2(input.x, input.y));
    const __half z = __float2half(input.z);
    return make_half3(xy.x, xy.y, z);
}