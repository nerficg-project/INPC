#pragma once

#include "helper_math.h"
#include "fp16_utils.h"
#include <cstdint>

namespace inpc::rasterization::bilinear::kernels {

    __global__ void preprocess_cu(
        const float3* positions,
        const half4* features,
        const float* opacities,
        const float4* w2c,
        const float3* cam_position,
        uint* primitive_n_touched_tiles,
        ushort4* primitive_screen_bounds,
        float2* primitive_screen_coords,
        float4* primitive_features_view,
        float* primitive_depth,
        uint* primitive_depth_keys,
        uint* primitive_indices,
        uint* n_visible_primitives,
        uint* n_instances,
        const uint n_primitives,
        const uint grid_width,
        const uint grid_height,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float near_plane,
        const float far_plane);

    __global__ void preprocess_backward_cu(
        const float3* positions,
        const float3* cam_position,
        const uint* primitive_n_touched_tiles,
        const float4* grad_features_view,
        half4* grad_features,
        const uint n_primitives);

    __global__ void preprocess_inference_cu(
        const float3* positions,
        const __half* features_raw,
        const float4* w2c,
        const float3* cam_position,
        uint* primitive_n_touched_tiles,
        ushort4* primitive_screen_bounds,
        float2* primitive_screen_coords,
        float4* primitive_features_view,
        float* primitive_depth,
        float* primitive_opacities,
        uint* primitive_depth_keys,
        uint* primitive_indices,
        uint* n_visible_primitives,
        uint* n_instances,
        const uint n_primitives,
        const uint grid_width,
        const uint grid_height,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float near_plane,
        const float far_plane);
    
    __global__ void apply_depth_ordering_cu(
        const uint* primitive_indices_sorted,
        const uint* primitive_n_touched_tiles,
        uint* primitive_offset,
        const uint n_visible_primitives);
    
    __global__ void create_instances_cu(
        const uint* primitive_indices_sorted,
        const uint* primitive_offsets,
        const ushort4* primitive_screen_bounds,
        ushort* instance_keys,
        uint* instance_primitive_indices,
        const uint grid_width,
        const uint n_visible_primitives);
    
    __global__ void extract_instance_ranges_cu(
        const ushort* instance_keys,
        uint2* tile_instance_ranges,
        const uint n_instances);

    __global__ void blend_forward_cu(
        const uint2* tile_instance_ranges,
        const uint* instance_primitive_indices,
        const float* opacities,
        const float2* primitive_screen_coords,
        const float4* primitive_features,
        float* feature_image,
        float* alpha_image,
        float* blending_weights,
        const uint width,
        const uint height,
        const uint grid_width);

    __global__ void blend_backward_cu(
        const uint2* tile_instance_ranges,
        const uint* instance_primitive_indices,
        const float* opacities,
        const float2* primitive_screen_coords,
        const float4* primitive_features,
        const float* grad_feature_image,
        const float* grad_alpha_image,
        const float* feature_image,
        const float* alpha_image,
        float4* grad_features,
        float* grad_opacities,
        const uint width,
        const uint height,
        const uint grid_width);

    __global__ void blend_inference_cu(
        const uint2* tile_instance_ranges,
        const uint* instance_primitive_indices,
        const float2* primitive_screen_coords,
        const float4* primitive_features,
        const float* primitive_depth,
        const float* primitive_opacity,
        float* feature_image,
        float* depth_image,
        float* alpha_image,
        const uint width,
        const uint height,
        const uint grid_width);

}
