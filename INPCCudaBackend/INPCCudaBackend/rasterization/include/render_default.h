#pragma once

#include "helper_math.h"
#include "fp16_utils.h"

namespace inpc::rasterization::inference {

    __global__ void preprocess_cu(
        const float4* w2c,
        const float3* cam_position,
        const float3* positions,
        const __half* features_raw,
        ulonglong4* fragment_keys_unsorted,
        uint4* fragment_point_indices_unsorted,
        float2* screen_coords,
        float4* features_view,
        float* opacities,
        uint* pixel_fragment_counts,
        const uint n_points,
        const uint width,
        const uint height,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float near_plane,
        const float far_plane);

    __global__ void blend_cu(
        const uint* fragment_offsets,
        const uint* fragment_point_indices,
        const float2* point_screen_coords,
        const float4* point_features_view,
        const float* point_opacities,
        const half4* bg_image,
        float* image,
        const uint n_pixels,
        const uint width,
        const float multisampling_factor);

}