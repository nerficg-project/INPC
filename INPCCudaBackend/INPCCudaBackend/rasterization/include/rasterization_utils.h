#pragma once

#include "helper_math.h"
#include "fp16_utils.h"

__host__ inline int extract_end_bit(uint32_t n) {
    int leading_zeros = 0;
    if ((n & 0xffff0000) == 0) { leading_zeros += 16; n <<= 16; }
    if ((n & 0xff000000) == 0) { leading_zeros += 8; n <<= 8; }
    if ((n & 0xf0000000) == 0) { leading_zeros += 4; n <<= 4; }
    if ((n & 0xc0000000) == 0) { leading_zeros += 2; n <<= 2; }
    if ((n & 0x80000000) == 0) { leading_zeros += 1; }
    return 32 - leading_zeros;
}

__device__ inline float4 convert_features(
	const half4* coefficients,
	const float3& cam_position,
	const float3& position_world)
{
	const auto [x, y, z] = normalize(position_world - cam_position);
	const float xx = x * x, yy = y * y, zz = z * z;
	const float xy = x * y, yz = y * z, xz = x * z;
	return 0.28209479177387814f * half42float4(coefficients[0])
		 - 0.48860251190291987f * y * half42float4(coefficients[1])
		 + 0.48860251190291987f * z * half42float4(coefficients[2])
		 - 0.48860251190291987f * x * half42float4(coefficients[3])
		 + 1.0925484305920792f * xy * half42float4(coefficients[4])
		 - 1.0925484305920792f * yz * half42float4(coefficients[5])
		 + (0.94617469575755997f * zz - 0.31539156525251999f) * half42float4(coefficients[6])
		 - 1.0925484305920792f * xz * half42float4(coefficients[7])
		 + (0.54627421529603959f * xx - 0.54627421529603959f * yy) * half42float4(coefficients[8]);
}

__device__ inline float4 convert_features(
	const __half* coefficients,
	const float3& cam_position,
	const float3& position_world)
{
	const auto [x, y, z] = normalize(position_world - cam_position);
	const float xx = x * x, yy = y * y, zz = z * z;
	const float xy = x * y, yz = y * z, xz = x * z;
	return 0.28209479177387814f * half42float4(make_half4(coefficients[0], coefficients[1], coefficients[2], coefficients[3]))
		 - 0.48860251190291987f * y * half42float4(make_half4(coefficients[4], coefficients[5], coefficients[6], coefficients[7]))
		 + 0.48860251190291987f * z * half42float4(make_half4(coefficients[8], coefficients[9], coefficients[10], coefficients[11]))
		 - 0.48860251190291987f * x * half42float4(make_half4(coefficients[12], coefficients[13], coefficients[14], coefficients[15]))
		 + 1.0925484305920792f * xy * half42float4(make_half4(coefficients[16], coefficients[17], coefficients[18], coefficients[19]))
		 - 1.0925484305920792f * yz * half42float4(make_half4(coefficients[20], coefficients[21], coefficients[22], coefficients[23]))
		 + (0.94617469575755997f * zz - 0.31539156525251999f) * half42float4(make_half4(coefficients[24], coefficients[25], coefficients[26], coefficients[27]))
		 - 1.0925484305920792f * xz * half42float4(make_half4(coefficients[28], coefficients[29], coefficients[30], coefficients[31]))
		 + (0.54627421529603959f * xx - 0.54627421529603959f * yy) * half42float4(make_half4(coefficients[32], coefficients[33], coefficients[34], coefficients[35]));
}
