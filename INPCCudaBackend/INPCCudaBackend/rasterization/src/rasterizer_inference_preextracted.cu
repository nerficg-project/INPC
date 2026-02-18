#include "rasterizer.h"
#include "rasterizer_config.h"
#include "utils.h"
#include "rasterization_utils.h"
#include "render_preextracted.h"
#include "helper_math.h"
#include "fp16_utils.h"
#include <functional>
#include <cub/cub.cuh>

namespace inference_config = config::inference;
namespace inference_preextracted = inpc::rasterization::inference_preextracted;

void inference_preextracted::render(
    std::function<char* (size_t)> per_point_buffers_func,
    std::function<char* (size_t)> per_fragment_buffers_func,
    std::function<char* (size_t)> per_pixel_buffers_func,
    const float4* w2c,
    const float3* cam_position,
    const float3* positions,
    const half4* features,
    const float* opacities,
    const half4* bg_image,
    float* image,
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

    static cudaStream_t memset_stream = 0;
    static bool streams_initialized = false;
    if (!streams_initialized && !inference_config::render::debug) {
        cudaStreamCreate(&memset_stream);
        streams_initialized = true;
    }
    cudaMemsetAsync(per_pixel_buffers.ranges, 0, sizeof(uint2) * n_pixels, memset_stream);

    preprocess_cu<<<div_round_up(n_points, inference_config::render_preextracted::block_size_preprocess), inference_config::render_preextracted::block_size_preprocess>>>(
        w2c,
        positions,
        per_point_buffers.depths,
        per_point_buffers.screen_coords,
        per_point_buffers.n_visible_fragments,
        n_points,
        width,
        height,
        fx,
        fy,
        cx,
        cy,
        near_plane,
        far_plane);
    CHECK_CUDA(inference_config::render_preextracted::debug, "preprocess_cu");

    cub::DeviceScan::InclusiveSum(
        per_point_buffers.cub_workspace,
        per_point_buffers.cub_workspace_size,
        per_point_buffers.n_visible_fragments,
        per_point_buffers.offsets,
        n_points);
    CHECK_CUDA(inference_config::render_preextracted::debug, "cub::DeviceScan::InclusiveSum");

    uint n_fragments;
    cudaMemcpy(&n_fragments, per_point_buffers.offsets + n_points - 1, sizeof(uint), cudaMemcpyDeviceToHost);
    char* per_fragment_buffers_blob = per_fragment_buffers_func(required<PerFragmentBuffers>(n_fragments));
    PerFragmentBuffers per_fragment_buffers = PerFragmentBuffers::from_blob(per_fragment_buffers_blob, n_fragments);

    create_fragments_cu<<<div_round_up(n_points, inference_config::render_preextracted::block_size_create_fragments), inference_config::render_preextracted::block_size_create_fragments>>>(
        per_point_buffers.depths,
        per_point_buffers.screen_coords,
        per_point_buffers.n_visible_fragments,
        per_point_buffers.offsets,
        per_fragment_buffers.fragment_keys.Current(),
        per_fragment_buffers.fragment_point_indices.Current(),
        n_points,
        width,
        height);
    CHECK_CUDA(inference_config::render_preextracted::debug, "create_fragments_cu");

    const int key_size = extract_end_bit(n_pixels - 1) + 32;
    cub::DeviceRadixSort::SortPairs(
        per_fragment_buffers.cub_workspace,
        per_fragment_buffers.cub_workspace_size,
        per_fragment_buffers.fragment_keys,
        per_fragment_buffers.fragment_point_indices,
        n_fragments,
        0,
        key_size);
    CHECK_CUDA(inference_config::render_preextracted::debug, "cub::DeviceRadixSort::SortPairs");

    if constexpr (!inference_config::render_preextracted::debug) {
        cudaStreamSynchronize(memset_stream);
    }

    if (n_fragments > 0) {
        find_ranges_cu<<<div_round_up(n_fragments, inference_config::render_preextracted::block_size_find_ranges), inference_config::render_preextracted::block_size_find_ranges>>>(
            per_fragment_buffers.fragment_keys.Current(),
            per_pixel_buffers.ranges,
            n_fragments);
        CHECK_CUDA(inference_config::render_preextracted::debug, "find_ranges_cu");
    }

    blend_cu<<<div_round_up(n_pixels * 32, inference_config::render_preextracted::block_size_blend), inference_config::render_preextracted::block_size_blend>>>(
        cam_position,
        per_pixel_buffers.ranges,
        per_fragment_buffers.fragment_point_indices.Current(),
        positions,
        per_point_buffers.screen_coords,
        features,
        opacities,
        bg_image,
        image,
        n_pixels,
        width);
    CHECK_CUDA(inference_config::render_preextracted::debug, "blend_cu");
}
