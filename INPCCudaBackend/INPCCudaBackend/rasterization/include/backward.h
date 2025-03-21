#pragma once

#include "helper_math.h"
#include "fp16_utils.h"

namespace inpc::rasterization::training {

    __global__ void blend_backward_cu(
        const uint* fragment_offsets,
        const uint* fragment_point_indices,
        const float2* point_screen_coords,
        const float4* point_features_view,
        const float* point_opacities,
        const float* image,
        const float* alpha,
        const float* grad_image,
        const float* grad_alpha,
        float4* grad_features_view,
        float* grad_opacities,
        const uint n_pixels,
        const uint width);

    __global__ void convert_features_backward_cu(
        const float3* cam_position,
        const float3* positions,
        const float4* grad_features_view,
        half4* grad_features,
        const uint n_points);

}