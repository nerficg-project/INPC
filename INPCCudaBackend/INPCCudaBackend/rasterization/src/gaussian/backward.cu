#include "gaussian/backward.h"
#include "gaussian/kernels.cuh"
#include "gaussian/buffer_utils.h"
#include "gaussian/config.h"
#include "rasterization_utils.h"
#include "utils.h"
#include "helper_math.h"

void inpc::rasterization::gaussian::backward(
    const float* grad_image,
    const float* grad_alpha,
    const float* image,
    const float* alpha,
    const float3* positions,
    const float* opacities,
    const float3* cam_position,
    char* per_primitive_buffers_blob,
    char* per_tile_buffers_blob,
    char* per_instance_buffers_blob,
    float4* grad_features_view,
    half4* grad_features,
    float* grad_opacities,
    const int n_primitives,
    const int width,
    const int height,
    const int n_instances,
    const int instance_primitive_indices_selector)
{
    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const dim3 block(config::tile_width, config::tile_height, 1);
    const int n_tiles = grid.x * grid.y;
    const int end_bit = extract_end_bit(n_tiles - 1);

    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives);
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);
    PerInstanceBuffers per_instance_buffers = PerInstanceBuffers::from_blob(per_instance_buffers_blob, n_instances, end_bit);
    per_instance_buffers.primitive_indices.selector = instance_primitive_indices_selector;

    kernels::blend_backward_cu<<<grid, block>>>(
        per_tile_buffers.instance_ranges,
        per_instance_buffers.primitive_indices.Current(),
        opacities,
        per_primitive_buffers.screen_coords,
        per_primitive_buffers.features_view,
        per_primitive_buffers.conic,
        grad_image,
        grad_alpha,
        image,
        alpha,
        grad_features_view,
        grad_opacities,
        width,
        height,
        grid.x
    );
    CHECK_CUDA(config::debug_backward, "blend_backward")

    kernels::preprocess_backward_cu<<<div_round_up(n_primitives, config::block_size_preprocess), config::block_size_preprocess>>>(
        positions,
        cam_position,
        per_primitive_buffers.n_touched_tiles,
        grad_features_view,
        grad_features,
        n_primitives
    );
    CHECK_CUDA(config::debug_backward, "preprocess_backward")

}
