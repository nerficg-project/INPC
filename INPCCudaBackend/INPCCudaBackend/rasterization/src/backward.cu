#include "backward.h"
#include "rasterizer_config.h"
#include "helper_math.h"
#include "fp16_utils.h"
#include "utils.h"

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
        const uint width)
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
        float4 features_pixel = make_float4(
            image[pixel_idx],
            image[pixel_idx + n_pixels],
            image[pixel_idx + 2 * n_pixels],
            image[pixel_idx + 3 * n_pixels]
        );
        const float4 grad_features_pixel = make_float4(
            grad_image[pixel_idx],
            grad_image[pixel_idx + n_pixels],
            grad_image[pixel_idx + 2 * n_pixels],
            grad_image[pixel_idx + 3 * n_pixels]
        );
        const float grad_from_alpha_partial = grad_alpha[pixel_idx] * (1.0f - alpha[pixel_idx]);

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

            const float blending_weight = transmittance * fragment_opacity;
            // TODO: is this even needed? if n_active < warp_size there will be no next loop iteration
            const bool compute_grads = blending_weight > 0.0f;
            if (compute_grads) {
                // after submission, we found that our feature gradients are incorrectly multiplied by the bilinear interpolation weights twice
                // interestingly, the correct gradients result in worse quality after training
                // this suggests that the quadratic falloff is beneficial for the training process, which is why we kept it as the default
                if constexpr (constexpr bool correct_feat_grads = false) {
                    atomicAdd(&grad_features_view[point_idx].x, blending_weight * grad_features_pixel.x);
                    atomicAdd(&grad_features_view[point_idx].y, blending_weight * grad_features_pixel.y);
                    atomicAdd(&grad_features_view[point_idx].z, blending_weight * grad_features_pixel.z);
                    atomicAdd(&grad_features_view[point_idx].w, blending_weight * grad_features_pixel.w);
                }
                else {
                    atomicAdd(&grad_features_view[point_idx].x, blending_weight * grad_features_pixel.x * blerp_weight);
                    atomicAdd(&grad_features_view[point_idx].y, blending_weight * grad_features_pixel.y * blerp_weight);
                    atomicAdd(&grad_features_view[point_idx].z, blending_weight * grad_features_pixel.z * blerp_weight);
                    atomicAdd(&grad_features_view[point_idx].w, blending_weight * grad_features_pixel.w * blerp_weight);
                }
            }

            // TODO: active mask could be made smaller during the loop
            const float4 features_lane = point_features_view[point_idx];
            const float4 weighted_features_lane = blending_weight * features_lane;
            float4 features_pixel_precomputed = features_pixel;
            for (int i = 0; i < n_active; ++i) {
                features_pixel_precomputed -= weighted_features_lane;
                if (lane_idx == i) features_pixel = features_pixel_precomputed;
                // TODO dont run this for the last lane
                features_pixel_precomputed.x = __shfl_up_sync(active_mask, features_pixel_precomputed.x, 1);
                features_pixel_precomputed.y = __shfl_up_sync(active_mask, features_pixel_precomputed.y, 1);
                features_pixel_precomputed.z = __shfl_up_sync(active_mask, features_pixel_precomputed.z, 1);
                features_pixel_precomputed.w = __shfl_up_sync(active_mask, features_pixel_precomputed.w, 1);
            }

            if (compute_grads) {
                const float used_transmittance_rcp = 1.0f / fmaxf(used_transmittance, FLT_EPS);
                const float grad_opacity_from_alpha = grad_from_alpha_partial * used_transmittance_rcp;
                const float grad_opacity_from_features = dot(features_lane * transmittance - features_pixel * used_transmittance_rcp, grad_features_pixel);
                const float grad_opacity_fragment = grad_opacity_from_alpha + grad_opacity_from_features;
                atomicAdd(&grad_opacities[point_idx], grad_opacity_fragment * blerp_weight);
            }

            const int last_lane = n_active - 1;
            transmittance = __shfl_sync(active_mask, transmittance * used_transmittance, last_lane);
            features_pixel.x = __shfl_sync(active_mask, features_pixel.x, last_lane);
            features_pixel.y = __shfl_sync(active_mask, features_pixel.y, last_lane);
            features_pixel.z = __shfl_sync(active_mask, features_pixel.z, last_lane);
            features_pixel.w = __shfl_sync(active_mask, features_pixel.w, last_lane);
        }
    }

    __global__ void convert_features_backward_cu(
        const float3* cam_position,
        const float3* positions,
        const float4* grad_features_view,
        half4* grad_features,
        const uint n_points)
    {
        const uint point_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (point_idx >= n_points) return;
        const float4 grads = grad_features_view[point_idx];
        constexpr int sh_coefficients_per_channel = 9;
        half4 *const grad_out = grad_features + point_idx * sh_coefficients_per_channel;
        const auto [x, y, z] = normalize(positions[point_idx] - cam_position[0]);
        const float xx = x * x, yy = y * y, zz = z * z;
        const float xy = x * y, yz = y * z, xz = x * z;
        grad_out[0] = float42half4(0.28209479177387814f * grads);
        grad_out[1] = float42half4(-0.48860251190291987f * y * grads);
        grad_out[2] = float42half4(0.48860251190291987f * z * grads);
        grad_out[3] = float42half4(-0.48860251190291987f * x * grads);
        grad_out[4] = float42half4(1.0925484305920792f * xy * grads);
        grad_out[5] = float42half4(-1.0925484305920792f * yz * grads);
        grad_out[6] = float42half4((0.94617469575755997f * zz - 0.31539156525251999f) * grads);
        grad_out[7] = float42half4(-1.0925484305920792f * xz * grads);
        grad_out[8] = float42half4((0.54627421529603959f * xx - 0.54627421529603959f * yy) * grads);
    }

}

