#include "render_preextracted.h"
#include "rasterizer_config.h"
#include "helper_math.h"
#include "fp16_utils.h"
#include "utils.h"
#include "rasterization_utils.h"
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
        const float far_plane)
    {
        const uint point_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (point_idx >= n_points) return;
        // projection and culling
        const float3 position_world = positions[point_idx];
        const float3 position_view = transform_point_affine(w2c, position_world);
        const float z = position_view.z;
        const float inv_z = 1.0f / fmaxf(z, FLT_EPS); // assume near_plane > FLT_EPS
        const float x = position_view.x * inv_z * fx + cx;
        const float y = position_view.y * inv_z * fy + cy;
        const float width_f = static_cast<float>(width);
        const float height_f = static_cast<float>(height);
        const bool is_visible = z >= near_plane && z <= far_plane && x >= -0.5f && x < width_f + 0.5f && y >= -0.5f && y < height_f + 0.5f;
        const bool instance_00_valid = is_visible && x >= 0.5f && y >= 0.5f;
        const bool instance_01_valid = is_visible && y >= 0.5f && x < width_f - 0.5f;
        const bool instance_10_valid = is_visible && x >= 0.5f && y < height_f - 0.5f;
        const bool instance_11_valid = is_visible && x < width_f - 0.5f && y < height_f - 0.5f;
        // write point info
        n_visible_fragments[point_idx] = static_cast<uint>(instance_00_valid) + static_cast<uint>(instance_01_valid) + static_cast<uint>(instance_10_valid) + static_cast<uint>(instance_11_valid);
        if (!is_visible) return;
        depths[point_idx] = z;
        screen_coords[point_idx] = make_float2(x, y);
    }

    __global__ void create_fragments_cu(
        const float* point_depths,
        const float2* point_screen_coords,
        const uint* point_n_visible_fragments,
        const uint* point_offsets,
        uint64_t* fragment_keys_unsorted,
        uint* fragment_point_indices_unsorted,
        const uint n_points,
        const uint width,
        const uint height)
    {
        const uint point_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (point_idx >= n_points || point_n_visible_fragments[point_idx] == 0) return;
        uint offset = point_idx == 0 ? 0 : point_offsets[point_idx - 1];
        const auto [x, y] = point_screen_coords[point_idx];
        // compute and write key/value pairs for each visible fragment
        const float width_f = static_cast<float>(width);
        const float height_f = static_cast<float>(height);
        const bool instance_00_valid = x >= 0.5f && y >= 0.5f;
        const bool instance_01_valid = y >= 0.5f && x < width_f - 0.5f;
        const bool instance_10_valid = x >= 0.5f && y < height_f - 0.5f;
        const bool instance_11_valid = x < width_f - 0.5f && y < height_f - 0.5f;
        const int pixel_idx_x = max(__float2int_rd(x - 0.5f), -1);
        const int pixel_idx_y = max(__float2int_rd(y - 0.5f), -1);
        const int pixel_idx_00 = pixel_idx_y * width + pixel_idx_x;
        const int pixel_idx_01 = pixel_idx_y * width + pixel_idx_x + 1;
        const int pixel_idx_10 = (pixel_idx_y + 1) * width + pixel_idx_x;
        const int pixel_idx_11 = (pixel_idx_y + 1) * width + pixel_idx_x + 1;
        const uint64_t depth_key = __float_as_uint(point_depths[point_idx]);
        if (instance_00_valid) {
            fragment_keys_unsorted[offset] = (static_cast<uint64_t>(pixel_idx_00) << 32) | depth_key;
            fragment_point_indices_unsorted[offset] = point_idx;
            ++offset;
        }
        if (instance_01_valid) {
            fragment_keys_unsorted[offset] = (static_cast<uint64_t>(pixel_idx_01) << 32) | depth_key;
            fragment_point_indices_unsorted[offset] = point_idx;
            ++offset;
        }
        if (instance_10_valid) {
            fragment_keys_unsorted[offset] = (static_cast<uint64_t>(pixel_idx_10) << 32) | depth_key;
            fragment_point_indices_unsorted[offset] = point_idx;
            ++offset;
        }
        if (instance_11_valid) {
            fragment_keys_unsorted[offset] = (static_cast<uint64_t>(pixel_idx_11) << 32) | depth_key;
            fragment_point_indices_unsorted[offset] = point_idx;
        }
    }

    __global__ void find_ranges_cu(
        const uint64_t* fragment_keys,
        uint2* ranges,
        const uint n_fragments)
    {
        const uint fragment_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (fragment_idx >= n_fragments) return;
        // read pixel index from key and update if at min/max index for pixel
        const uint64_t key = fragment_keys[fragment_idx];
        const uint pixel = key >> 32;
        if (fragment_idx == 0) ranges[pixel].x = 0;
        else {
            const uint prev_pixel = fragment_keys[fragment_idx - 1u] >> 32u;
            if (pixel != prev_pixel) {
                ranges[prev_pixel].y = fragment_idx;
                ranges[pixel].x = fragment_idx;
            }
        }
        if (fragment_idx == n_fragments - 1) ranges[pixel].y = n_fragments;
    }

    __global__ void blend_cu(
        const float3* cam_position,
        const uint2* ranges,
        const uint* fragment_point_indices,
        const float3* point_positions,
        const float2* point_screen_coords,
        const half4* point_features,
        const float* point_opacities,
        const half4* bg_image,
        float* image,
        const uint n_pixels,
        const uint width)
    {
        constexpr uint warp_size = 32;
        constexpr uint warp_mask = warp_size - 1;
        const uint thread_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        const uint pixel_idx = thread_idx / warp_size;
        if (pixel_idx >= n_pixels) return;

        const float2 pixel = make_float2(make_uint2(pixel_idx % width, pixel_idx / width)) + 0.5f;

        const auto [start_idx, end_idx] = ranges[pixel_idx];
        const uint lane_idx = thread_idx & warp_mask;
        const bool leader = lane_idx == 0;

        float transmittance = 1.0f;
        float4 features_pixel = make_float4(0.0f);

        for (uint current_base_idx = start_idx; current_base_idx < end_idx && transmittance >= config::transmittance_threshold; current_base_idx += warp_size) {
            const uint n_active = min(end_idx - current_base_idx, warp_size);
            const uint current_data_idx = current_base_idx + lane_idx;
            if (current_data_idx >= end_idx && !leader) return;

            const uint point_idx = fragment_point_indices[current_data_idx];
            const float2 blerp_factors = 1.0f - fabsf(point_screen_coords[point_idx] - pixel);
            const float blerp_weight = __saturatef(blerp_factors.x) * __saturatef(blerp_factors.y);
            const float fragment_opacity = point_opacities[point_idx] * blerp_weight;
            const float used_transmittance = 1.0f - fragment_opacity;

            // TODO: active mask could be made smaller during the loop
            const uint active_mask = n_active == 32 ? 0xffffffff : (1 << n_active) - 1;
            float transmittance_precomputed = transmittance;
            for (int i = 1; i <= n_active; ++i) {
                transmittance_precomputed = __shfl_up_sync(active_mask, transmittance_precomputed * used_transmittance, 1);
                if (lane_idx == i) transmittance = transmittance_precomputed < config::transmittance_threshold ? 0.0f : transmittance_precomputed;
            }

            constexpr int sh_coefficients_per_channel = 9;
            const float4 point_features_view = convert_features(point_features + point_idx * sh_coefficients_per_channel, cam_position[0], point_positions[point_idx]);
            // TODO: could use butterfly reduction here
            float4 weighted_features_lane = transmittance * fragment_opacity * point_features_view;
            for (int i = 0; i < n_active; ++i) {
                if (leader) features_pixel += weighted_features_lane;
                weighted_features_lane.x = __shfl_down_sync(active_mask, weighted_features_lane.x, 1);
                weighted_features_lane.y = __shfl_down_sync(active_mask, weighted_features_lane.y, 1);
                weighted_features_lane.z = __shfl_down_sync(active_mask, weighted_features_lane.z, 1);
                weighted_features_lane.w = __shfl_down_sync(active_mask, weighted_features_lane.w, 1);
            }
            const int last_lane = n_active - 1;
            transmittance = __shfl_sync(active_mask, transmittance * used_transmittance, last_lane);
        }
        if (leader) {
            const float4 final_features = transmittance < config::transmittance_threshold ? features_pixel : features_pixel + transmittance * half42float4(bg_image[pixel_idx]);
            image[pixel_idx] = final_features.x;
            image[pixel_idx + n_pixels] = final_features.y;
            image[pixel_idx + 2 * n_pixels] = final_features.z;
            image[pixel_idx + 3 * n_pixels] = final_features.w;
        }
    }

}
