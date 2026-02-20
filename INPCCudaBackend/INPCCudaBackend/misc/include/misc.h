#pragma once

#include "helper_math.h"

namespace inpc::misc {

    void compute_normalized_weight_decay_grads(
        const float* weights,
        float* grads,
        const uint n_weights);

    void add_normalized_weight_decay_grads(
        const float* weights,
        float* grads,
        const uint n_weights);

    void spherical_contraction(
        const float3* positions,
        float3* positions_contracted,
        const uint n_points);

    void cauchy_loss(
        const float* input,
        const float* target,
        float* elementwise_loss,
        const uint n_elements);

    void cauchy_loss_backward(
        const float* grad,
        const float* input,
        const float* target,
        float* grad_input,
        const uint n_elements);

}
