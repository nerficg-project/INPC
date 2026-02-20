#pragma once

#include "helper_math.h"
#include "fp16_utils.h"

namespace inpc::rasterization::bilinear {

    void backward(
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
        const int instance_primitive_indices_selector);

}
