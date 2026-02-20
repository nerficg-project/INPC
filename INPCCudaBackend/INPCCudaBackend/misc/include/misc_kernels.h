#pragma once

#include "helper_math.h"

namespace inpc::misc {

    template <bool add>
    __global__ void compute_normalized_weight_decay_grads_cu(
        const float* weights,
        float* grads,
        const uint n_weights);

    __global__ void spherical_contraction_cu(
        const float3* positions,
        float3* positions_contracted,
        const uint n_points);

    __global__ void cauchy_loss_cu(
        const float* input,
        const float* target,
        float* elementwise_loss,
        const uint n_elements);

    __global__ void cauchy_loss_backward_cu(
        const float* grad,
        const float* input,
        const float* target,
        float* grad_input,
        const uint n_elements);

}
