#include "render_default.h"
#include "rasterizer_config.h"
#include "helper_math.h"
#include "fp16_utils.h"
#include "utils.h"
#include "rasterization_utils.h"
#include <cstdint>

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
        // compute and write key/value pairs for each fragment
        const bool instance_00_valid = is_visible && x >= 0.5f && y >= 0.5f;
        const bool instance_01_valid = is_visible && y >= 0.5f && x < width_f - 0.5f;
        const bool instance_10_valid = is_visible && x >= 0.5f && y < height_f - 0.5f;
        const bool instance_11_valid = is_visible && x < width_f - 0.5f && y < height_f - 0.5f;
        const int pixel_idx_x = max(__float2int_rd(x - 0.5f), -1);
        const int pixel_idx_y = max(__float2int_rd(y - 0.5f), -1);
        const int pixel_idx_00 = pixel_idx_y * width + pixel_idx_x;
        const int pixel_idx_01 = pixel_idx_y * width + pixel_idx_x + 1;
        const int pixel_idx_10 = (pixel_idx_y + 1) * width + pixel_idx_x;
        const int pixel_idx_11 = (pixel_idx_y + 1) * width + pixel_idx_x + 1;
        const uint64_t depth_key = __float_as_uint(z);
        fragment_keys_unsorted[point_idx] = make_ulonglong4(
            instance_00_valid ? (static_cast<uint64_t>(pixel_idx_00) << 32) | depth_key : 0xffffffffffffffffu,
            instance_01_valid ? (static_cast<uint64_t>(pixel_idx_01) << 32) | depth_key : 0xffffffffffffffffu,
            instance_10_valid ? (static_cast<uint64_t>(pixel_idx_10) << 32) | depth_key : 0xffffffffffffffffu,
            instance_11_valid ? (static_cast<uint64_t>(pixel_idx_11) << 32) | depth_key : 0xffffffffffffffffu
        );
        if (!is_visible) return;
        fragment_point_indices_unsorted[point_idx] = make_uint4(point_idx, point_idx, point_idx, point_idx);
        // write screen coords and converted sh features
        screen_coords[point_idx] = make_float2(x, y);
        const __half* point_features_raw = features_raw + point_idx * config::inference::render::features_per_point;
        opacities[point_idx] = 1.0f - __expf(-__expf(__half2float(point_features_raw[0])));
        features_view[point_idx] = convert_features(point_features_raw + 1, cam_position[0], position_world);
        // increment per-pixel fragment count
        if (instance_00_valid) atomicAdd(&pixel_fragment_counts[pixel_idx_00], 1);
        if (instance_01_valid) atomicAdd(&pixel_fragment_counts[pixel_idx_01], 1);
        if (instance_10_valid) atomicAdd(&pixel_fragment_counts[pixel_idx_10], 1);
        if (instance_11_valid) atomicAdd(&pixel_fragment_counts[pixel_idx_11], 1);
    }

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
        const float multisampling_factor)
    {
        constexpr uint warp_size = 32;
        constexpr uint warp_mask = warp_size - 1;
        const uint thread_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        const uint pixel_idx = thread_idx / warp_size;
        if (pixel_idx >= n_pixels) return;

        const float2 pixel = make_float2(make_uint2(pixel_idx % width, pixel_idx / width)) + 0.5f;

        const uint start_idx = pixel_idx == 0 ? 0 : fragment_offsets[pixel_idx - 1];
        const uint end_idx = fragment_offsets[pixel_idx];
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

            // TODO: could use butterfly reduction here
            float4 weighted_features_lane = transmittance * fragment_opacity * point_features_view[point_idx];
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
            image[pixel_idx] += final_features.x * multisampling_factor;
            image[pixel_idx + n_pixels] += final_features.y * multisampling_factor;
            image[pixel_idx + 2 * n_pixels] += final_features.z * multisampling_factor;
            image[pixel_idx + 3 * n_pixels] += final_features.w * multisampling_factor;
        }
    }

}
