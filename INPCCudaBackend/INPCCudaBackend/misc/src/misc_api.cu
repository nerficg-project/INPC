#include "misc_api.h"
#include "misc.h"
#include "helper_math.h"

namespace cuda_api = inpc::misc;

torch::Tensor cuda_api::compute_normalized_weight_decay_grads_wrapper(
    const torch::Tensor& weights)
{
    const uint n_weights = weights.size(0);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    torch::Tensor grads = torch::empty({n_weights}, float_options);

    compute_normalized_weight_decay_grads(
        weights.contiguous().data_ptr<float>(),
        grads.contiguous().data_ptr<float>(),
        n_weights);

    return grads;
}

void cuda_api::add_normalized_weight_decay_grads_wrapper(
    const torch::Tensor& weights,
    torch::Tensor& grads)
{
    const uint n_weights = weights.size(0);

    add_normalized_weight_decay_grads(
        weights.data_ptr<float>(),
        grads.data_ptr<float>(),
        n_weights);
}

torch::Tensor cuda_api::spherical_contraction_wrapper(
    const torch::Tensor& positions)
{
    const uint n_points = positions.size(0);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    torch::Tensor positions_contracted = torch::empty({n_points, 3}, float_options);

    if (n_points == 0) return positions_contracted;

    spherical_contraction(
        reinterpret_cast<float3*>(positions.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(positions_contracted.contiguous().data_ptr<float>()),
        n_points);

    return positions_contracted;
}

torch::Tensor cuda_api::cauchy_loss_wrapper(
    const torch::Tensor& input,
    const torch::Tensor& target)
{
    const uint n_elements = input.numel();
    torch::Tensor elementwise_loss = torch::empty_like(input);

    cauchy_loss(
        input.contiguous().data_ptr<float>(),
        target.contiguous().data_ptr<float>(),
        elementwise_loss.data_ptr<float>(),
        n_elements);

    return elementwise_loss.sum();
}

torch::Tensor cuda_api::cauchy_loss_backward_wrapper(
    const torch::Tensor& grad,
    const torch::Tensor& input,
    const torch::Tensor& target)
{
    const uint n_elements = input.numel();
    torch::Tensor grad_input = torch::empty_like(input);

    cauchy_loss_backward(
        grad.contiguous().data_ptr<float>(),
        input.contiguous().data_ptr<float>(),
        target.contiguous().data_ptr<float>(),
        grad_input.data_ptr<float>(),
        n_elements);

    return grad_input;
}
