#pragma once

#include "helper_math.h"
#include "fp16_utils.h"
#include <functional>

namespace inpc::rasterization::bilinear {

    void inference_preextracted(
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
        const float far_plane);

}
