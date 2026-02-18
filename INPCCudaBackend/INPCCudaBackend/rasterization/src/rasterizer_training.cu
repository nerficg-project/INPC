#include "rasterizer.h"
#include "rasterizer_config.h"
#include "utils.h"
#include "rasterization_utils.h"
#include "forward.h"
#include "backward.h"
#include "helper_math.h"
#include "fp16_utils.h"
#include <functional>
#include <cub/cub.cuh>

namespace forward_config = config::training::forward;
namespace backward_config = config::training::backward;
namespace training = inpc::rasterization::training;

int training::forward(
    std::function<char* (size_t)> per_point_buffers_func,
    std::function<char* (size_t)> per_pixel_buffers_func,
    const float4* w2c,
    const float3* cam_position,
    const float3* positions,
    const half4* features,
    const float* opacities,
    float* image,
    float* alpha,
    float* blending_weights,
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
    char* per_point_buffers_blob = per_point_buffers_func(required<PerPointBuffers>(n_points));
    PerPointBuffers per_point_buffers = PerPointBuffers::from_blob(per_point_buffers_blob, n_points);

    const uint n_pixels = width * height;
    char* per_pixel_buffers_blob = per_pixel_buffers_func(required<PerPixelBuffers>(n_pixels));
    PerPixelBuffers per_pixel_buffers = PerPixelBuffers::from_blob(per_pixel_buffers_blob, n_pixels);

    cudaMemset(per_pixel_buffers.fragment_counts, 0, sizeof(uint) * n_pixels);

    preprocess_cu<<<div_round_up(n_points, forward_config::block_size_preprocess), forward_config::block_size_preprocess>>>(
        w2c,
        cam_position,
        positions,
        features,
        reinterpret_cast<ulonglong4*>(per_point_buffers.fragment_keys.Current()),
        reinterpret_cast<uint4*>(per_point_buffers.fragment_point_indices.Current()),
        per_point_buffers.screen_coords,
        per_point_buffers.features_view,
        per_pixel_buffers.fragment_counts,
        n_points,
        width,
        height,
        fx,
        fy,
        cx,
        cy,
        near_plane,
        far_plane);
    CHECK_CUDA(forward_config::debug, "preprocess_cu");

    static cudaStream_t sum_stream = 0;
    static cudaStream_t sort_stream = 0;
    static bool streams_initialized = false;
    if (!streams_initialized && !forward_config::debug) {
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
    CHECK_CUDA(forward_config::debug, "cub::DeviceScan::InclusiveSum");

    const int key_size = extract_end_bit(n_pixels - 1) + 32;
    cub::DeviceRadixSort::SortPairs(
        per_point_buffers.cub_workspace,
        per_point_buffers.cub_workspace_size,
        per_point_buffers.fragment_keys,
        per_point_buffers.fragment_point_indices,
        n_points * 4,
        0,
        key_size,
        sort_stream);
    CHECK_CUDA(forward_config::debug, "cub::DeviceRadixSort::SortPairs");

    if constexpr (!forward_config::debug) {
        cudaStreamSynchronize(sum_stream);
        cudaStreamSynchronize(sort_stream);
    }

    blend_cu<<<div_round_up(n_pixels * 32, forward_config::block_size_blend), forward_config::block_size_blend>>>(
        per_pixel_buffers.fragment_offsets,
        per_point_buffers.fragment_point_indices.Current(),
        per_point_buffers.screen_coords,
        per_point_buffers.features_view,
        opacities,
        image,
        alpha,
        blending_weights,
        n_pixels,
        width);
    CHECK_CUDA(forward_config::debug, "blend_cu");

    return per_point_buffers.fragment_point_indices.selector;
}

void training::backward(
    const float* grad_image,
    const float* grad_alpha,
    const float* image,
    const float* alpha,
    const float3* positions,
    const float* opacities,
    const float3* cam_position,
    char* per_point_buffers_blob,
    char* per_pixel_buffers_blob,
    float4* grad_features_view,
    half4* grad_features,
    float* grad_opacities,
    const uint n_points,
    const uint n_pixels,
    const uint width,
    const int fragment_point_indices_selector)
{
    PerPointBuffers per_point_buffers = PerPointBuffers::from_blob(per_point_buffers_blob, n_points);
    per_point_buffers.fragment_point_indices.selector = fragment_point_indices_selector;
    PerPixelBuffers per_pixel_buffers = PerPixelBuffers::from_blob(per_pixel_buffers_blob, n_pixels);

    blend_backward_cu<<<div_round_up(n_pixels * 32, backward_config::block_size_blend), backward_config::block_size_blend>>>(
        per_pixel_buffers.fragment_offsets,
        per_point_buffers.fragment_point_indices.Current(),
        per_point_buffers.screen_coords,
        per_point_buffers.features_view,
        opacities,
        image,
        alpha,
        grad_image,
        grad_alpha,
        grad_features_view,
        grad_opacities,
        n_pixels,
        width);
    CHECK_CUDA(backward_config::debug, "blend_backward_cu");

    convert_features_backward_cu<<<div_round_up(n_points, backward_config::block_size_convert_features), backward_config::block_size_convert_features>>>(
        cam_position,
        positions,
        grad_features_view,
        grad_features,
        n_points);
    CHECK_CUDA(backward_config::debug, "convert_features_backward_cu");
}