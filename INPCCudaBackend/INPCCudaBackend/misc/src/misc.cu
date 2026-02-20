#include "misc.h"
#include "misc_config.h"
#include "misc_kernels.h"
#include "helper_math.h"
#include "utils.h"

namespace cuda_api = inpc::misc;

void cuda_api::compute_normalized_weight_decay_grads(
    const float* weights,
    float* grads,
    const uint n_weights)
{
    compute_normalized_weight_decay_grads_cu<false><<<div_round_up(n_weights, config::block_size_compute_normalized_weight_decay_grads), config::block_size_compute_normalized_weight_decay_grads>>>(
        weights,
        grads,
        n_weights);
    CHECK_CUDA(config::debug, "compute_normalized_weight_decay_grads_cu (set)");
}

void cuda_api::add_normalized_weight_decay_grads(
    const float* weights,
    float* grads,
    const uint n_weights)
{
    compute_normalized_weight_decay_grads_cu<true><<<div_round_up(n_weights, config::block_size_add_normalized_weight_decay_grads), config::block_size_add_normalized_weight_decay_grads>>>(
        weights,
        grads,
        n_weights);
    CHECK_CUDA(config::debug, "compute_normalized_weight_decay_grads_cu (add)");
}

void cuda_api::spherical_contraction(
    const float3* positions,
    float3* positions_contracted,
    const uint n_points)
{
    spherical_contraction_cu<<<div_round_up(n_points, config::block_size_spherical_contraction), config::block_size_spherical_contraction>>>(
        positions,
        positions_contracted,
        n_points);
    CHECK_CUDA(config::debug, "spherical_contraction_cu");
}

void cuda_api::cauchy_loss(
    const float* input,
    const float* target,
    float* elementwise_loss,
    const uint n_elements)
{
    cauchy_loss_cu<<<div_round_up(n_elements, config::block_size_cauchy_loss), config::block_size_cauchy_loss>>>(
        input,
        target,
        elementwise_loss,
        n_elements);
    CHECK_CUDA(config::debug, "cauchy_loss_cu");
}

void cuda_api::cauchy_loss_backward(
    const float* grad,
    const float* input,
    const float* target,
    float* grad_input,
    const uint n_elements)
{
    cauchy_loss_backward_cu<<<div_round_up(n_elements, config::block_size_cauchy_loss_backward), config::block_size_cauchy_loss_backward>>>(
        grad,
        input,
        target,
        grad_input,
        n_elements);
    CHECK_CUDA(config::debug, "cauchy_loss_backward_cu");
}
