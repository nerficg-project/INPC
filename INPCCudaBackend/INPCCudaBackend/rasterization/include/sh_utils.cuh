#pragma once

#include "helper_math.h"
#include "fp16_utils.h"

__forceinline__ __device__ float4 convert_features(
        const float3& position_world,
        const float3& cam_position,
        const half4* features)
{
    const auto [x, y, z] = normalize(position_world - cam_position);
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, yz = y * z, xz = x * z;
    return 0.28209479177387814f * half42float4(features[0])
        - 0.48860251190291987f * y * half42float4(features[1])
        + 0.48860251190291987f * z * half42float4(features[2])
        - 0.48860251190291987f * x * half42float4(features[3])
        + 1.0925484305920792f * xy * half42float4(features[4])
        - 1.0925484305920792f * yz * half42float4(features[5])
        + (0.94617469575755997f * zz - 0.31539156525251999f) * half42float4(features[6])
        - 1.0925484305920792f * xz * half42float4(features[7])
        + (0.54627421529603959f * xx - 0.54627421529603959f * yy) * half42float4(features[8]);
}

__forceinline__ __device__ float4 convert_features(
    const float3& position_world,
    const float3& cam_position,
    const __half* features)
{
    const auto [x, y, z] = normalize(position_world - cam_position);
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, yz = y * z, xz = x * z;
    return 0.28209479177387814f * half42float4(make_half4(features[0], features[1], features[2], features[3]))
        - 0.48860251190291987f * y * half42float4(make_half4(features[4], features[5], features[6], features[7]))
        + 0.48860251190291987f * z * half42float4(make_half4(features[8], features[9], features[10], features[11]))
        - 0.48860251190291987f * x * half42float4(make_half4(features[12], features[13], features[14], features[15]))
        + 1.0925484305920792f * xy * half42float4(make_half4(features[16], features[17], features[18], features[19]))
        - 1.0925484305920792f * yz * half42float4(make_half4(features[20], features[21], features[22], features[23]))
        + (0.94617469575755997f * zz - 0.31539156525251999f) * half42float4(make_half4(features[24], features[25], features[26], features[27]))
        - 1.0925484305920792f * xz * half42float4(make_half4(features[28], features[29], features[30], features[31]))
        + (0.54627421529603959f * xx - 0.54627421529603959f * yy) * half42float4(make_half4(features[32], features[33], features[34], features[35]));
}

__forceinline__ __device__ void convert_features_backward(
    const float3& position_world,
    const float3& cam_position,
    const float4& grad_features_view,
    half4* grad_features)
{
    const auto [x, y, z] = normalize(position_world - cam_position);
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, yz = y * z, xz = x * z;
    grad_features[0] = float42half4(0.28209479177387814f * grad_features_view);
    grad_features[1] = float42half4(-0.48860251190291987f * y * grad_features_view);
    grad_features[2] = float42half4(0.48860251190291987f * z * grad_features_view);
    grad_features[3] = float42half4(-0.48860251190291987f * x * grad_features_view);
    grad_features[4] = float42half4(1.0925484305920792f * xy * grad_features_view);
    grad_features[5] = float42half4(-1.0925484305920792f * yz * grad_features_view);
    grad_features[6] = float42half4((0.94617469575755997f * zz - 0.31539156525251999f) * grad_features_view);
    grad_features[7] = float42half4(-1.0925484305920792f * xz * grad_features_view);
    grad_features[8] = float42half4((0.54627421529603959f * xx - 0.54627421529603959f * yy) * grad_features_view);
}
