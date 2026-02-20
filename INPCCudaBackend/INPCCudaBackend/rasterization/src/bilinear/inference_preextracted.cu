#include "bilinear/inference_preextracted.h"
#include "bilinear/kernels.cuh"
#include "bilinear/buffer_utils.h"
#include "bilinear/config.h"
#include "rasterization_utils.h"
#include "utils.h"
#include "helper_math.h"
#include <cub/cub.cuh>
#include <functional>

void inpc::rasterization::bilinear::inference_preextracted(
    std::function<char* (size_t)> per_primitive_buffers_func,
    std::function<char* (size_t)> per_tile_buffers_func,
    std::function<char* (size_t)> per_instance_buffers_func,
    const float3* positions,
    const half4* features,
    const float* opacities,
    const float4* w2c,
    const float3* cam_position,
    float* image,
    float* depth,
    float* alpha,
    const int n_primitives,
    const int width,
    const int height,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const float near_plane,
    const float far_plane)
{
    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const dim3 block(config::tile_width, config::tile_height, 1);
    const int n_tiles = grid.x * grid.y;
    const int end_bit = extract_end_bit(n_tiles - 1);

    char* per_primitive_buffers_blob = per_primitive_buffers_func(required<PerPrimitiveBuffers>(n_primitives));
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives);

    char* per_tile_buffers_blob = per_tile_buffers_func(required<PerTileBuffers>(n_tiles));
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);

    static cudaStream_t memset_stream = 0;
    if constexpr (!config::debug_inference_preextracted) {
        static bool memset_stream_initialized = false;
        if (!memset_stream_initialized) {
            cudaStreamCreate(&memset_stream);
            memset_stream_initialized = true;
        }
        cudaMemsetAsync(per_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles, memset_stream);
    }
    else cudaMemset(per_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles);

    cudaMemset(per_primitive_buffers.n_visible_primitives, 0, sizeof(uint));
    cudaMemset(per_primitive_buffers.n_instances, 0, sizeof(uint));

    kernels::preprocess_cu<<<div_round_up(n_primitives, config::block_size_preprocess), config::block_size_preprocess>>>(
        positions,
        features,
        opacities,
        w2c,
        cam_position,
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.screen_bounds,
        per_primitive_buffers.screen_coords,
        per_primitive_buffers.features_view,
        per_primitive_buffers.depth,
        per_primitive_buffers.depth_keys.Current(),
        per_primitive_buffers.primitive_indices.Current(),
        per_primitive_buffers.n_visible_primitives,
        per_primitive_buffers.n_instances,
        n_primitives,
        grid.x,
        grid.y,
        fx,
        fy,
        cx,
        cy,
        near_plane,
        far_plane
    );
    CHECK_CUDA(config::debug_inference_preextracted, "preprocess")

    int n_visible_primitives;
    cudaMemcpy(&n_visible_primitives, per_primitive_buffers.n_visible_primitives, sizeof(uint), cudaMemcpyDeviceToHost);
    int n_instances;
    cudaMemcpy(&n_instances, per_primitive_buffers.n_instances, sizeof(uint), cudaMemcpyDeviceToHost);

    cub::DeviceRadixSort::SortPairs(
        per_primitive_buffers.cub_workspace,
        per_primitive_buffers.cub_workspace_size,
        per_primitive_buffers.depth_keys,
        per_primitive_buffers.primitive_indices,
        n_visible_primitives,
        0, 32
    );
    CHECK_CUDA(config::debug_inference_preextracted, "cub::DeviceRadixSort::SortPairs (Depth)")

    kernels::apply_depth_ordering_cu<<<div_round_up(n_visible_primitives, config::block_size_create_instances), config::block_size_create_instances>>>(
        per_primitive_buffers.primitive_indices.Current(),
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.offset,
        n_visible_primitives
    );
    CHECK_CUDA(config::debug_inference_preextracted, "apply_depth_ordering")

    cub::DeviceScan::ExclusiveSum(
        per_primitive_buffers.cub_workspace,
        per_primitive_buffers.cub_workspace_size,
        per_primitive_buffers.offset,
        per_primitive_buffers.offset,
        n_visible_primitives
    );
    CHECK_CUDA(config::debug_inference_preextracted, "cub::DeviceScan::ExclusiveSum")

    char* per_instance_buffers_blob = per_instance_buffers_func(required<PerInstanceBuffers>(n_instances, end_bit));
    PerInstanceBuffers per_instance_buffers = PerInstanceBuffers::from_blob(per_instance_buffers_blob, n_instances, end_bit);

    kernels::create_instances_cu<<<div_round_up(n_visible_primitives, config::block_size_create_instances), config::block_size_create_instances>>>(
        per_primitive_buffers.primitive_indices.Current(),
        per_primitive_buffers.offset,
        per_primitive_buffers.screen_bounds,
        per_instance_buffers.keys.Current(),
        per_instance_buffers.primitive_indices.Current(),
        grid.x,
        n_visible_primitives
    );
    CHECK_CUDA(config::debug_inference_preextracted, "create_instances")

    cub::DeviceRadixSort::SortPairs(
        per_instance_buffers.cub_workspace,
        per_instance_buffers.cub_workspace_size,
        per_instance_buffers.keys,
        per_instance_buffers.primitive_indices,
        n_instances,
        0, end_bit
    );
    CHECK_CUDA(config::debug_inference_preextracted, "cub::DeviceRadixSort::SortPairs (Tile)")

    if constexpr (!config::debug_inference_preextracted) cudaStreamSynchronize(memset_stream);

    if (n_instances > 0) {
        kernels::extract_instance_ranges_cu<<<div_round_up(n_instances, config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges>>>(
            per_instance_buffers.keys.Current(),
            per_tile_buffers.instance_ranges,
            n_instances
        );
        CHECK_CUDA(config::debug_inference_preextracted, "extract_instance_ranges")
    }

    kernels::blend_inference_cu<<<grid, block>>>(
        per_tile_buffers.instance_ranges,
        per_instance_buffers.primitive_indices.Current(),
        per_primitive_buffers.screen_coords,
        per_primitive_buffers.features_view,
        per_primitive_buffers.depth,
        opacities,
        image,
        depth,
        alpha,
        width,
        height,
        grid.x
    );
    CHECK_CUDA(config::debug_inference_preextracted, "blend_inference")

}
