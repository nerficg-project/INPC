#pragma once

#include "helper_math.h"
#include "fp16_utils.h"
#include <cstdint>

namespace inpc::rasterization::inference_preextracted {

    __global__ void preprocess_cu(
        const float4* w2c,
        const float3* positions,
        float* depths,
        float2* screen_coords,
        uint* n_visible_fragments,
        const uint n_points,
        const uint width,
        const uint height,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float near_plane,
        const float far_plane);

    __global__ void create_fragments_cu(
        const float* point_depths,
        const float2* point_screen_coords,
        const uint* point_n_visible_fragments,
        const uint* point_offsets,
        uint64_t* fragment_keys_unsorted,
        uint* fragment_point_indices_unsorted,
        const uint n_points,
        const uint width,
        const uint height);

    __global__ void find_ranges_cu(
        const uint64_t* fragment_keys,
        uint2* ranges,
        const uint n_fragments);

    __global__ void blend_cu(
        const float3* cam_poistion,
        const uint2* ranges,
        const uint* fragment_point_indices,
        const float3* point_positions,
        const float2* point_screen_coords,
        const half4* point_features,
        const float* point_opacities,
        const half4* bg_image,
        float* image,
        const uint n_pixels,
        const uint width);

}