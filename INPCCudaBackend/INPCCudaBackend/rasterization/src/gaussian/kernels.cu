#include "gaussian/kernels.cuh"
#include "helper_math.h"
#include "fp16_utils.h"
#include "sh_utils.cuh"
#include "gaussian/config.h"
#include <cstdint>
#include <cooperative_groups.h>

namespace inpc::rasterization::gaussian::kernels {

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
        float3* primitive_conic,
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
        const float far_plane)
    {
        const uint primitive_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (primitive_idx >= n_primitives) return;

        primitive_n_touched_tiles[primitive_idx] = 0;

        const float3 position_world = positions[primitive_idx];
        const float4 w2c_r3 = w2c[2];
        const float depth = w2c_r3.x * position_world.x + w2c_r3.y * position_world.y + w2c_r3.z * position_world.z + w2c_r3.w;
        if (depth < near_plane || depth > far_plane) return;

        const float4 w2c_r1 = w2c[0];
        const float4 w2c_r2 = w2c[1];
        const float x = (w2c_r1.x * position_world.x + w2c_r1.y * position_world.y + w2c_r1.z * position_world.z + w2c_r1.w) / depth;
        const float y = (w2c_r2.x * position_world.x + w2c_r2.y * position_world.y + w2c_r2.z * position_world.z + w2c_r2.w) / depth;
        // TODO: distortion
        const float2 screen_coords = make_float2(
            x * fx + cx,
            y * fy + cy
        );

        constexpr float target_sigma_screen = 0.3f;
        constexpr float target_depth = 0.5f;
        constexpr float variance_world_numerator = target_sigma_screen * target_sigma_screen * target_depth * target_depth;
        const float variance_world = variance_world_numerator / (fx * fy);
        const float J_00 = fx / depth;
        const float J_02 = -J_00 * clamp(x, -1.3f * cx / fx, 1.3f * cx / fx);
        const float J_11 = fy / depth;
        const float J_12 = -J_11 * clamp(y, -1.3f * cy / fy, 1.3f * cy / fy);
        float3 cov2d = variance_world * make_float3(
            J_00 * J_00 + J_02 * J_02,
            J_02 * J_12,
            J_11 * J_11 + J_12 * J_12
        );
        cov2d.x += config::dilation;
        cov2d.z += config::dilation;
        const float determinant = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
        if (determinant == 0.0f) return;
        const float determinant_rcp = 1.0f / determinant;
        const float3 conic = make_float3(
            cov2d.z * determinant_rcp,
            -cov2d.y * determinant_rcp,
            cov2d.x * determinant_rcp
        );
        float extent_x = 3.0f * sqrtf(cov2d.x) - 0.5f;
        float extent_y = 3.0f * sqrtf(cov2d.z) - 0.5f;

        const uint4 screen_bounds = make_uint4(
            min(grid_width, static_cast<uint>(max(0, __float2int_rd((screen_coords.x - extent_x) / static_cast<float>(config::tile_width))))), // x_min
            min(grid_width, static_cast<uint>(max(0, __float2int_ru((screen_coords.x + extent_x) / static_cast<float>(config::tile_width))))), // x_max
            min(grid_height, static_cast<uint>(max(0, __float2int_rd((screen_coords.y - extent_y) / static_cast<float>(config::tile_height))))), // y_min
            min(grid_height, static_cast<uint>(max(0, __float2int_ru((screen_coords.y + extent_y) / static_cast<float>(config::tile_height))))) // y_max
        );

        const uint n_touched_tiles = (screen_bounds.y - screen_bounds.x) * (screen_bounds.w - screen_bounds.z);
        if (n_touched_tiles == 0) return;

        primitive_n_touched_tiles[primitive_idx] = n_touched_tiles;
        primitive_screen_bounds[primitive_idx] = make_ushort4(
            static_cast<ushort>(screen_bounds.x),
            static_cast<ushort>(screen_bounds.y),
            static_cast<ushort>(screen_bounds.z),
            static_cast<ushort>(screen_bounds.w)
        );
        primitive_screen_coords[primitive_idx] = screen_coords;
        constexpr int features_per_channel = 9;
        primitive_features_view[primitive_idx] = convert_features(position_world, cam_position[0], features + primitive_idx * features_per_channel);
        primitive_depth[primitive_idx] = depth;
        primitive_conic[primitive_idx] = conic;

        const uint offset = atomicAdd(n_visible_primitives, 1);
        const uint depth_key = __float_as_uint(depth);
        primitive_depth_keys[offset] = depth_key;
        primitive_indices[offset] = primitive_idx;
        atomicAdd(n_instances, n_touched_tiles);
    }

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
        float3* primitive_conic,
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
        const float far_plane,
        const float sigma_world,
        const float sigma_cutoff)
    {
        const uint primitive_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (primitive_idx >= n_primitives) return;

        primitive_n_touched_tiles[primitive_idx] = 0;

        const float3 position_world = positions[primitive_idx];
        const float4 w2c_r3 = w2c[2];
        const float depth = w2c_r3.x * position_world.x + w2c_r3.y * position_world.y + w2c_r3.z * position_world.z + w2c_r3.w;
        if (depth < near_plane || depth > far_plane) return;

        constexpr int features_per_primitive = 37;
        const __half* features_raw_ptr = features_raw + primitive_idx * features_per_primitive;

        const float opacity = 1.0f - __expf(-__expf(__half2float(features_raw_ptr[0])));

        const float4 w2c_r1 = w2c[0];
        const float4 w2c_r2 = w2c[1];
        const float x = (w2c_r1.x * position_world.x + w2c_r1.y * position_world.y + w2c_r1.z * position_world.z + w2c_r1.w) / depth;
        const float y = (w2c_r2.x * position_world.x + w2c_r2.y * position_world.y + w2c_r2.z * position_world.z + w2c_r2.w) / depth;
        // TODO: distortion
        const float2 screen_coords = make_float2(
            x * fx + cx,
            y * fy + cy
        );

        const float variance_world = sigma_world * sigma_world;
        const float J_00 = fx / depth;
        const float J_02 = -J_00 * clamp(x, -1.3f * cx / fx, 1.3f * cx / fx);
        const float J_11 = fy / depth;
        const float J_12 = -J_11 * clamp(y, -1.3f * cy / fy, 1.3f * cy / fy);
        float3 cov2d = variance_world * make_float3(
            J_00 * J_00 + J_02 * J_02,
            J_02 * J_12,
            J_11 * J_11 + J_12 * J_12
        );
        cov2d.x += config::dilation;
        cov2d.z += config::dilation;
        const float determinant = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
        if (determinant == 0.0f) return;
        const float determinant_rcp = 1.0f / determinant;
        const float3 conic = make_float3(
            cov2d.z * determinant_rcp,
            -cov2d.y * determinant_rcp,
            cov2d.x * determinant_rcp
        );
        float extent_x = sigma_cutoff * sqrtf(cov2d.x) - 0.5f;
        float extent_y = sigma_cutoff * sqrtf(cov2d.z) - 0.5f;

        const uint4 screen_bounds = make_uint4(
            min(grid_width, static_cast<uint>(max(0, __float2int_rd((screen_coords.x - extent_x) / static_cast<float>(config::tile_width))))), // x_min
            min(grid_width, static_cast<uint>(max(0, __float2int_ru((screen_coords.x + extent_x) / static_cast<float>(config::tile_width))))), // x_max
            min(grid_height, static_cast<uint>(max(0, __float2int_rd((screen_coords.y - extent_y) / static_cast<float>(config::tile_height))))), // y_min
            min(grid_height, static_cast<uint>(max(0, __float2int_ru((screen_coords.y + extent_y) / static_cast<float>(config::tile_height))))) // y_max
        );

        const uint n_touched_tiles = (screen_bounds.y - screen_bounds.x) * (screen_bounds.w - screen_bounds.z);
        if (n_touched_tiles == 0) return;

        primitive_n_touched_tiles[primitive_idx] = n_touched_tiles;
        primitive_screen_bounds[primitive_idx] = make_ushort4(
            static_cast<ushort>(screen_bounds.x),
            static_cast<ushort>(screen_bounds.y),
            static_cast<ushort>(screen_bounds.z),
            static_cast<ushort>(screen_bounds.w)
        );
        primitive_screen_coords[primitive_idx] = screen_coords;
        primitive_features_view[primitive_idx] = convert_features(position_world, cam_position[0], features_raw_ptr + 1);
        primitive_depth[primitive_idx] = depth;
        primitive_opacities[primitive_idx] = opacity;
        primitive_conic[primitive_idx] = conic;

        const uint offset = atomicAdd(n_visible_primitives, 1);
        const uint depth_key = __float_as_uint(depth);
        primitive_depth_keys[offset] = depth_key;
        primitive_indices[offset] = primitive_idx;
        atomicAdd(n_instances, n_touched_tiles);
    }

    __global__ void preprocess_preextracted_cu(
        const float3* positions,
        const half4* features,
        const float* opacities,
        const float* scales,
        const float4* w2c,
        const float3* cam_position,
        uint* primitive_n_touched_tiles,
        ushort4* primitive_screen_bounds,
        float2* primitive_screen_coords,
        float4* primitive_features_view,
        float* primitive_depth,
        float3* primitive_conic,
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
        const float far_plane,
        const float sigma_world,
        const float sigma_cutoff)
    {
        const uint primitive_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (primitive_idx >= n_primitives) return;

        primitive_n_touched_tiles[primitive_idx] = 0;

        const float3 position_world = positions[primitive_idx];
        const float4 w2c_r3 = w2c[2];
        const float depth = w2c_r3.x * position_world.x + w2c_r3.y * position_world.y + w2c_r3.z * position_world.z + w2c_r3.w;
        if (depth < near_plane || depth > far_plane) return;

        const float4 w2c_r1 = w2c[0];
        const float4 w2c_r2 = w2c[1];
        const float x = (w2c_r1.x * position_world.x + w2c_r1.y * position_world.y + w2c_r1.z * position_world.z + w2c_r1.w) / depth;
        const float y = (w2c_r2.x * position_world.x + w2c_r2.y * position_world.y + w2c_r2.z * position_world.z + w2c_r2.w) / depth;
        // TODO: distortion
        const float2 screen_coords = make_float2(
            x * fx + cx,
            y * fy + cy
        );

        float variance_world;
        if (scales == nullptr) variance_world = sigma_world * sigma_world;
        else {
            const float scale = scales[primitive_idx];
            variance_world = scale * scale;
        }
        const float J_00 = fx / depth;
        const float J_02 = -J_00 * clamp(x, -1.3f * cx / fx, 1.3f * cx / fx);
        const float J_11 = fy / depth;
        const float J_12 = -J_11 * clamp(y, -1.3f * cy / fy, 1.3f * cy / fy);
        float3 cov2d = variance_world * make_float3(
            J_00 * J_00 + J_02 * J_02,
            J_02 * J_12,
            J_11 * J_11 + J_12 * J_12
        );
        cov2d.x += config::dilation;
        cov2d.z += config::dilation;
        const float determinant = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
        if (determinant == 0.0f) return;
        const float determinant_rcp = 1.0f / determinant;
        const float3 conic = make_float3(
            cov2d.z * determinant_rcp,
            -cov2d.y * determinant_rcp,
            cov2d.x * determinant_rcp
        );
        float extent_x = sigma_cutoff * sqrtf(cov2d.x) - 0.5f;
        float extent_y = sigma_cutoff * sqrtf(cov2d.z) - 0.5f;

        const uint4 screen_bounds = make_uint4(
            min(grid_width, static_cast<uint>(max(0, __float2int_rd((screen_coords.x - extent_x) / static_cast<float>(config::tile_width))))), // x_min
            min(grid_width, static_cast<uint>(max(0, __float2int_ru((screen_coords.x + extent_x) / static_cast<float>(config::tile_width))))), // x_max
            min(grid_height, static_cast<uint>(max(0, __float2int_rd((screen_coords.y - extent_y) / static_cast<float>(config::tile_height))))), // y_min
            min(grid_height, static_cast<uint>(max(0, __float2int_ru((screen_coords.y + extent_y) / static_cast<float>(config::tile_height))))) // y_max
        );

        const uint n_touched_tiles = (screen_bounds.y - screen_bounds.x) * (screen_bounds.w - screen_bounds.z);
        if (n_touched_tiles == 0) return;

        primitive_n_touched_tiles[primitive_idx] = n_touched_tiles;
        primitive_screen_bounds[primitive_idx] = make_ushort4(
            static_cast<ushort>(screen_bounds.x),
            static_cast<ushort>(screen_bounds.y),
            static_cast<ushort>(screen_bounds.z),
            static_cast<ushort>(screen_bounds.w)
        );
        primitive_screen_coords[primitive_idx] = screen_coords;
        constexpr int features_per_channel = 9;
        primitive_features_view[primitive_idx] = convert_features(position_world, cam_position[0], features + primitive_idx * features_per_channel);
        primitive_depth[primitive_idx] = depth;
        primitive_conic[primitive_idx] = conic;

        const uint offset = atomicAdd(n_visible_primitives, 1);
        const uint depth_key = __float_as_uint(depth);
        primitive_depth_keys[offset] = depth_key;
        primitive_indices[offset] = primitive_idx;
        atomicAdd(n_instances, n_touched_tiles);
    }

    __global__ void preprocess_backward_cu(
        const float3* positions,
        const float3* cam_position,
        const uint* primitive_n_touched_tiles,
        const float4* grad_features_view,
        half4* grad_features,
        const uint n_primitives)
    {
        const uint primitive_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0) return;

        constexpr int features_per_channel = 9;
        convert_features_backward(
            positions[primitive_idx],
            cam_position[0],
            grad_features_view[primitive_idx],
            grad_features + primitive_idx * features_per_channel
        );
    }

    __global__ void apply_depth_ordering_cu(
        const uint* primitive_indices_sorted,
        const uint* primitive_n_touched_tiles,
        uint* primitive_offset,
        const uint n_visible_primitives)
    {
        const uint idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (idx >= n_visible_primitives) return;
        const uint primitive_idx = primitive_indices_sorted[idx];
        primitive_offset[idx] = primitive_n_touched_tiles[primitive_idx];
    }

    __global__ void create_instances_cu(
        const uint* primitive_indices_sorted,
        const uint* primitive_offsets,
        const ushort4* primitive_screen_bounds,
        ushort* instance_keys,
        uint* instance_primitive_indices,
        const uint grid_width,
        const uint n_visible_primitives)
    {
        const uint idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (idx >= n_visible_primitives) return;
        const uint primitive_idx = primitive_indices_sorted[idx];
        const ushort4 screen_bounds = primitive_screen_bounds[primitive_idx];
        uint offset = primitive_offsets[idx];
        for (ushort y = screen_bounds.z; y < screen_bounds.w; ++y) {
            for (ushort x = screen_bounds.x; x < screen_bounds.y; ++x) {
                const ushort tile_idx = y * grid_width + x;
                instance_keys[offset] = tile_idx;
                instance_primitive_indices[offset] = primitive_idx;
                offset++;
            }
        }
    }

    __global__ void extract_instance_ranges_cu(
        const ushort* instance_keys,
        uint2* tile_instance_ranges,
        const uint n_instances)
    {
        const uint instance_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (instance_idx >= n_instances) return;
        const ushort instance_tile_idx = instance_keys[instance_idx];
        if (instance_idx == 0) tile_instance_ranges[instance_tile_idx].x = 0;
        else {
            const ushort previous_instance_tile_idx = instance_keys[instance_idx - 1];
            if (instance_tile_idx != previous_instance_tile_idx) {
                tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;
                tile_instance_ranges[instance_tile_idx].x = instance_idx;
            }
        }
        if (instance_idx == n_instances - 1) tile_instance_ranges[instance_tile_idx].y = n_instances;
    }

    __global__ void __launch_bounds__(config::block_size_blend) blend_forward_cu(
        const uint2* tile_instance_ranges,
        const uint* instance_primitive_indices,
        const float* opacities,
        const float2* primitive_screen_coords,
        const float4* primitive_features,
        const float3* primitive_conic,
        float* feature_image,
        float* alpha_image,
        float* blending_weights,
        const uint width,
        const uint height,
        const uint grid_width)
    {
        const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        const dim3 group_index = block.group_index();
        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        const uint2 pixel_coords = make_uint2(group_index.x * config::tile_width + thread_index.x, group_index.y * config::tile_height + thread_index.y);
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
        // setup shared memory
        __shared__ uint collected_primitive_indices[config::block_size_blend];
        __shared__ float collected_opacity[config::block_size_blend];
        __shared__ float2 collected_screen_coords[config::block_size_blend];
        __shared__ float4 collected_features[config::block_size_blend];
        __shared__ float3 collected_conic[config::block_size_blend];
        // initialize local storage
        float4 features_pixel = make_float4(0.0f);
        float transmittance = 1.0f;
        bool done = !inside;
        // collaborative loading and processing
        const uint2 tile_range = tile_instance_ranges[group_index.y * grid_width + group_index.x];
        for (int n_points_remaining = tile_range.y - tile_range.x, current_fetch_idx = tile_range.x + thread_rank; n_points_remaining > 0; n_points_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
            if (__syncthreads_count(done) == config::block_size_blend) break;
            if (current_fetch_idx < tile_range.y) {
                const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
                collected_primitive_indices[thread_rank] = primitive_idx;
                collected_opacity[thread_rank] = opacities[primitive_idx];
                collected_screen_coords[thread_rank] = primitive_screen_coords[primitive_idx];
                collected_features[thread_rank] = primitive_features[primitive_idx];
                collected_conic[thread_rank] = primitive_conic[primitive_idx];
            }
            block.sync();
            const int current_batch_size = min(config::block_size_blend, n_points_remaining);
            for (int j = 0; !done && j < current_batch_size; ++j) {
                const float2 distance = collected_screen_coords[j] - pixel;
                const float3 conic = collected_conic[j];
                const float exponent = -0.5f * (conic.x * distance.x * distance.x + conic.z * distance.y * distance.y) - conic.y * distance.x * distance.y;
                if (exponent > 0.0f) continue;
                const float G = expf(exponent);
                if (G < 0.01110899653f) continue;
                const float alpha = collected_opacity[j] * G;

                const float blending_weight = transmittance * alpha;
                if (blending_weight > 0.0f) atomicAdd(&blending_weights[collected_primitive_indices[j]], G * blending_weight);

                features_pixel += blending_weight * collected_features[j];
                transmittance *= 1.0f - alpha;

                if (transmittance < config::transmittance_threshold) done = true;
            }
        }
        if (inside) {
            const int pixel_idx = width * pixel_coords.y + pixel_coords.x;
            const int n_pixels = width * height;
            // store results
            feature_image[pixel_idx] = features_pixel.x;
            feature_image[n_pixels + pixel_idx] = features_pixel.y;
            feature_image[2 * n_pixels + pixel_idx] = features_pixel.z;
            feature_image[3 * n_pixels + pixel_idx] = features_pixel.w;
            alpha_image[pixel_idx] = transmittance < config::transmittance_threshold ? 1.0f : 1.0f - transmittance;
        }
    }

    __global__ void __launch_bounds__(config::block_size_blend) blend_backward_cu(
        const uint2* tile_instance_ranges,
        const uint* instance_primitive_indices,
        const float* opacities,
        const float2* primitive_screen_coords,
        const float4* primitive_features,
        const float3* primitive_conic,
        const float* grad_feature_image,
        const float* grad_alpha_image,
        const float* feature_image,
        const float* alpha_image,
        float4* grad_features,
        float* grad_opacities,
        const uint width,
        const uint height,
        const uint grid_width)
    {
        const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        const dim3 group_index = block.group_index();
        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        const uint2 pixel_coords = make_uint2(group_index.x * config::tile_width + thread_index.x, group_index.y * config::tile_height + thread_index.y);
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
        // setup shared memory
        __shared__ uint collected_primitive_indices[config::block_size_blend];
        __shared__ float collected_opacity[config::block_size_blend];
        __shared__ float2 collected_screen_coords[config::block_size_blend];
        __shared__ float4 collected_features[config::block_size_blend];
        __shared__ float3 collected_conic[config::block_size_blend];
        // initialize local storage
        float4 grad_features_pixel, features_pixel;
        float dL_dalpha_from_alpha_common;
        if (inside) {
            const int pixel_idx = width * pixel_coords.y + pixel_coords.x;
            const int n_pixels = width * height;
            grad_features_pixel = make_float4(
                grad_feature_image[pixel_idx],
                grad_feature_image[n_pixels + pixel_idx],
                grad_feature_image[2 * n_pixels + pixel_idx],
                grad_feature_image[3 * n_pixels + pixel_idx]
            );
            features_pixel = make_float4(
                feature_image[pixel_idx],
                feature_image[n_pixels + pixel_idx],
                feature_image[2 * n_pixels + pixel_idx],
                feature_image[3 * n_pixels + pixel_idx]
            );
            dL_dalpha_from_alpha_common = grad_alpha_image[pixel_idx] * (1.0f - alpha_image[pixel_idx]);
        }
        float4 features_pixel_new = make_float4(0.0f);
        float transmittance = 1.0f;
        bool done = !inside;
        // collaborative loading and processing
        const uint2 tile_range = tile_instance_ranges[group_index.y * grid_width + group_index.x];
        for (int n_points_remaining = tile_range.y - tile_range.x, current_fetch_idx = tile_range.x + thread_rank; n_points_remaining > 0; n_points_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
            if (__syncthreads_count(done) == config::block_size_blend) break;
            if (current_fetch_idx < tile_range.y) {
                const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
                collected_primitive_indices[thread_rank] = primitive_idx;
                collected_opacity[thread_rank] = opacities[primitive_idx];
                collected_screen_coords[thread_rank] = primitive_screen_coords[primitive_idx];
                collected_features[thread_rank] = primitive_features[primitive_idx];
                collected_conic[thread_rank] = primitive_conic[primitive_idx];
            }
            block.sync();
            const int current_batch_size = min(config::block_size_blend, n_points_remaining);
            for (int j = 0; !done && j < current_batch_size; ++j) {
                const float2 distance = collected_screen_coords[j] - pixel;
                const float3 conic = collected_conic[j];
                const float exponent = -0.5f * (conic.x * distance.x * distance.x + conic.z * distance.y * distance.y) - conic.y * distance.x * distance.y;
                if (exponent > 0.0f) continue;
                const float G = expf(exponent);
                if (G < 0.01110899653f) continue;
                const float alpha = collected_opacity[j] * G;

                const float blending_weight = transmittance * alpha;
                const uint primitive_idx = collected_primitive_indices[j];

                // feature gradients
                if (blending_weight > 0.0f) {
                    const float4 dL_dfeatures = blending_weight * grad_features_pixel;
                    atomicAdd(&grad_features[primitive_idx].x, dL_dfeatures.x);
                    atomicAdd(&grad_features[primitive_idx].y, dL_dfeatures.y);
                    atomicAdd(&grad_features[primitive_idx].z, dL_dfeatures.z);
                    atomicAdd(&grad_features[primitive_idx].w, dL_dfeatures.w);
                }

                // opacity gradient
                const float4 features = collected_features[j];
                features_pixel_new += blending_weight * features;
                const float one_minus_alpha = 1.0f - alpha;
                const float one_minus_alpha_rcp = 1.0f / fmaxf(one_minus_alpha, config::one_minus_alpha_eps);
                const float dL_dalpha_from_features = dot(transmittance * features + (features_pixel_new - features_pixel) * one_minus_alpha_rcp, grad_features_pixel);
                const float dL_dalpha_from_alpha = dL_dalpha_from_alpha_common * one_minus_alpha_rcp;
                const float dL_dalpha = dL_dalpha_from_features + dL_dalpha_from_alpha;
                const float dL_dopacity = dL_dalpha * G;
                if (dL_dopacity != 0.0f) atomicAdd(&grad_opacities[primitive_idx], dL_dopacity);

                transmittance *= one_minus_alpha;
                if (transmittance < config::transmittance_threshold) done = true;
            }
        }
    }

    __global__ void __launch_bounds__(config::block_size_blend) blend_inference_cu(
        const uint2* tile_instance_ranges,
        const uint* instance_primitive_indices,
        const float2* primitive_screen_coords,
        const float4* primitive_features,
        const float* primitive_depth,
        const float3* primitive_conic,
        const float* primitive_opacity,
        float* feature_image,
        float* depth_image,
        float* alpha_image,
        const uint width,
        const uint height,
        const uint grid_width,
        const float gaussian_threshold)
    {
        const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        const dim3 group_index = block.group_index();
        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        const uint2 pixel_coords = make_uint2(group_index.x * config::tile_width + thread_index.x, group_index.y * config::tile_height + thread_index.y);
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
        // setup shared memory
        __shared__ float collected_opacity[config::block_size_blend];
        __shared__ float2 collected_screen_coords[config::block_size_blend];
        __shared__ float4 collected_features[config::block_size_blend];
        __shared__ float collected_depth[config::block_size_blend];
        __shared__ float3 collected_conic[config::block_size_blend];
        // initialize local storage
        float4 features_pixel = make_float4(0.0f);
        float depth_pixel = 0.0f;
        float transmittance = 1.0f;
        bool done = !inside;
        // collaborative loading and processing
        const uint2 tile_range = tile_instance_ranges[group_index.y * grid_width + group_index.x];
        for (int n_points_remaining = tile_range.y - tile_range.x, current_fetch_idx = tile_range.x + thread_rank; n_points_remaining > 0; n_points_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
            if (__syncthreads_count(done) == config::block_size_blend) break;
            if (current_fetch_idx < tile_range.y) {
                const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
                collected_opacity[thread_rank] = primitive_opacity[primitive_idx];
                collected_screen_coords[thread_rank] = primitive_screen_coords[primitive_idx];
                collected_features[thread_rank] = primitive_features[primitive_idx];
                collected_depth[thread_rank] = primitive_depth[primitive_idx];
                collected_conic[thread_rank] = primitive_conic[primitive_idx];
            }
            block.sync();
            const int current_batch_size = min(config::block_size_blend, n_points_remaining);
            for (int j = 0; !done && j < current_batch_size; ++j) {
                const float2 distance = collected_screen_coords[j] - pixel;
                const float3 conic = collected_conic[j];
                const float exponent = -0.5f * (conic.x * distance.x * distance.x + conic.z * distance.y * distance.y) - conic.y * distance.x * distance.y;
                if (exponent > 0.0f) continue;
                const float G = __expf(exponent);
                if (G < gaussian_threshold) continue;
                const float alpha = collected_opacity[j] * G;

                const float blending_weight = transmittance * alpha;

                features_pixel += blending_weight * collected_features[j];
                depth_pixel += blending_weight * collected_depth[j];
                transmittance *= 1.0f - alpha;

                if (transmittance < config::transmittance_threshold) done = true;
            }
        }
        if (inside) {
            const float alpha_pixel = 1.0f - transmittance;
            depth_pixel = (alpha_pixel > 0.0f) ? depth_pixel / alpha_pixel : 0.0f;
            const int pixel_idx = width * pixel_coords.y + pixel_coords.x;
            const int n_pixels = width * height;
            // store results
            feature_image[pixel_idx] = features_pixel.x;
            feature_image[n_pixels + pixel_idx] = features_pixel.y;
            feature_image[2 * n_pixels + pixel_idx] = features_pixel.z;
            feature_image[3 * n_pixels + pixel_idx] = features_pixel.w;
            depth_image[pixel_idx] = depth_pixel;
            alpha_image[pixel_idx] = transmittance < config::transmittance_threshold ? 1.0f : alpha_pixel;
        }
    }

}
