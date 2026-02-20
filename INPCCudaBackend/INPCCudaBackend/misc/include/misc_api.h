#pragma once

#include <torch/extension.h>

namespace inpc::misc {

    torch::Tensor compute_normalized_weight_decay_grads_wrapper(
        const torch::Tensor& weights);

    void add_normalized_weight_decay_grads_wrapper(
        const torch::Tensor& weights,
        torch::Tensor& grads);

    torch::Tensor spherical_contraction_wrapper(
        const torch::Tensor& positions);

    torch::Tensor cauchy_loss_wrapper(
        const torch::Tensor& input,
        const torch::Tensor& target);

    torch::Tensor cauchy_loss_backward_wrapper(
        const torch::Tensor& grad,
        const torch::Tensor& input,
        const torch::Tensor& target);

}
