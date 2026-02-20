#include "misc_kernels.h"
#include "helper_math.h"

namespace inpc::misc {

    template <bool add>
    __global__ void compute_normalized_weight_decay_grads_cu(
        const float* weights,
        float* grads,
        const uint n_weights)
    {
        const uint weight_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (weight_idx >= n_weights) return;

        constexpr float f0 = 0.1f * 2.0f / float(210917376 - 177362944);
        constexpr float f1 = 0.1f * 2.0f / float(177362944 - 143808512);
        constexpr float f2 = 0.1f * 2.0f / float(143808512 - 110254080);
        constexpr float f3 = 0.1f * 2.0f / float(110254080 - 76699648);
        constexpr float f4 = 0.1f * 2.0f / float(76699648 - 43145216);
        constexpr float f5 = 0.1f * 2.0f / float(43145216 - 9590784);
        constexpr float f6 = 0.1f * 2.0f / float(9590784 - 1202176);
        constexpr float f7 = 0.1f * 2.0f / float(1202176 - 153600);
        constexpr float f8 = 0.1f * 2.0f / float(153600 - 22528);
        constexpr float f9 = 0.1f * 2.0f / float(22528 - 6144);

        float factor = 0.0f;
        if (weight_idx >= 177362944)      factor = f0;
        else if (weight_idx >= 143808512) factor = f1;
        else if (weight_idx >= 110254080) factor = f2;
        else if (weight_idx >= 76699648)  factor = f3;
        else if (weight_idx >= 43145216)  factor = f4;
        else if (weight_idx >= 9590784)   factor = f5;
        else if (weight_idx >= 1202176)   factor = f6;
        else if (weight_idx >= 153600)    factor = f7;
        else if (weight_idx >= 22528)     factor = f8;
        else if (weight_idx >= 6144)      factor = f9;

        if constexpr (add)
            grads[weight_idx] += factor * weights[weight_idx];
        else
            grads[weight_idx] = factor * weights[weight_idx];
    }

    template __global__ void compute_normalized_weight_decay_grads_cu<true>(
        const float*, float*, const uint);
    template __global__ void compute_normalized_weight_decay_grads_cu<false>(
        const float*, float*, const uint);

    __global__ void spherical_contraction_cu(
        const float3* positions,
        float3* positions_contracted,
        const uint n_points)
    {
        const uint point_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (point_idx >= n_points) return;

        const float3 position = positions[point_idx];
        const float length_sq = fmaxf(dot(position, position), 1.0f);
        const float scale = (sqrtf(length_sq) * 2.0f - 1.0f) / length_sq * 0.25f;
        const float3 contracted_position = scale * position + 0.5f;
        positions_contracted[point_idx] = make_float3(
            __saturatef(contracted_position.x),
            __saturatef(contracted_position.y),
            __saturatef(contracted_position.z)
        );
    }

    __global__ void cauchy_loss_cu(
        const float* input,
        const float* target,
        float* elementwise_loss,
        const uint n_elements)
    {
        const uint idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (idx >= n_elements) return;

        const float diff = input[idx] - target[idx];
        elementwise_loss[idx] = log1pf(diff * diff * 12.5f) / n_elements;
    }

    __global__ void cauchy_loss_backward_cu(
        const float* grad,
        const float* input,
        const float* target,
        float* grad_input,
        const uint n_elements)
    {
        const uint idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (idx >= n_elements) return;

        const float diff = input[idx] - target[idx];
        const float denominator = n_elements * (12.5f * diff * diff + 1.0f);
        grad_input[idx] = 25.0f * diff / fmaxf(denominator, 1.0e-12f) * grad[0];
    }

}
