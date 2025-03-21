#include "rasterizer.h"
#include "rasterizer_config.h"
#include "utils.h"
#include "rasterization_utils.h"
#include "render_default.h"
#include "helper_math.h"
#include "fp16_utils.h"
#include <functional>
#include <cub/cub.cuh>

namespace inference_config = config::inference;
namespace inference = inpc::rasterization::inference;
namespace training = inpc::rasterization::training;

void inference::render(
    std::function<char* (size_t)> per_point_buffers_func,
    std::function<char* (size_t)> per_pixel_buffers_func,
    const float4* w2c,
    const float3* cam_position,
    const float3* positions,
    const __half* features_raw,
    const half4* bg_image,
    float* image,
    const uint n_points,
    const uint width,
    const uint height,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const float near,
    const float far,
    const uint n_multisamples)
{
    const uint n_points_per_multisample = n_points / n_multisamples;
    const uint n_points_used = n_points_per_multisample * n_multisamples;
    const uint n_fragments = n_points_per_multisample * 4;
    const float multisampling_factor = 1.0f / static_cast<float>(n_multisamples);
    char* per_point_buffers_blob = per_point_buffers_func(required<PerPointBuffers>(n_points_used));
    PerPointBuffers per_point_buffers = PerPointBuffers::from_blob(per_point_buffers_blob, n_points_used);

    const uint n_pixels = width * height;
    const int key_size = extract_end_bit(n_pixels) + 32;
    char* per_pixel_buffers_blob = per_pixel_buffers_func(required<training::PerPixelBuffers>(n_pixels));
    training::PerPixelBuffers per_pixel_buffers = training::PerPixelBuffers::from_blob(per_pixel_buffers_blob, n_pixels);

    for (int i = 0; i < n_multisamples; ++i) {
        const uint buffer_offset = i * n_points_per_multisample;
        const float3* current_positions = positions + buffer_offset;
        const __half* current_features_raw = features_raw + buffer_offset * inference_config::render::features_per_point;

        cudaMemset(per_pixel_buffers.fragment_counts, 0, sizeof(uint) * n_pixels);

        preprocess_cu<<<div_round_up(n_points_per_multisample, inference_config::render::block_size_preprocess), inference_config::render::block_size_preprocess>>>(
            w2c,
            cam_position,
            current_positions,
            current_features_raw,
            reinterpret_cast<ulonglong4*>(per_point_buffers.fragment_keys.Current()),
            reinterpret_cast<uint4*>(per_point_buffers.fragment_point_indices.Current()),
            per_point_buffers.screen_coords,
            per_point_buffers.features_view,
            per_point_buffers.opacities,
            per_pixel_buffers.fragment_counts,
            n_points_per_multisample,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            near,
            far);
        CHECK_CUDA(inference_config::render::debug, "preprocess_cu");

        static cudaStream_t sum_stream = 0;
        static cudaStream_t sort_stream = 0;
        static bool streams_initialized = false;
        if (!streams_initialized && !inference_config::render::debug) {
            cudaStreamCreate(&sum_stream);
            cudaStreamCreate(&sort_stream);
            streams_initialized = true;
        }
        cub::DeviceScan::InclusiveSum(
            per_pixel_buffers.cub_workspace,
            per_pixel_buffers.cub_workspace_size,
            per_pixel_buffers.fragment_counts,
            per_pixel_buffers.fragment_offsets,
            n_pixels,
            sum_stream);
        CHECK_CUDA(inference_config::render::debug, "cub::DeviceScan::InclusiveSum");

        cub::DeviceRadixSort::SortPairs(
            per_point_buffers.cub_workspace,
            per_point_buffers.cub_workspace_size,
            per_point_buffers.fragment_keys,
            per_point_buffers.fragment_point_indices,
            n_fragments,
            0,
            key_size,
            sort_stream);
        CHECK_CUDA(inference_config::render::debug, "cub::DeviceRadixSort::SortPairs");

        if constexpr (!inference_config::render::debug) {
            cudaStreamSynchronize(sum_stream);
            cudaStreamSynchronize(sort_stream);
        }

        blend_cu<<<div_round_up(n_pixels * 32, inference_config::render::block_size_blend), inference_config::render::block_size_blend>>>(
            per_pixel_buffers.fragment_offsets,
            per_point_buffers.fragment_point_indices.Current(),
            per_point_buffers.screen_coords,
            per_point_buffers.features_view,
            per_point_buffers.opacities,
            bg_image,
            image,
            n_pixels,
            width,
            multisampling_factor);
        CHECK_CUDA(inference_config::render::debug, "blend_cu");
    }
}
