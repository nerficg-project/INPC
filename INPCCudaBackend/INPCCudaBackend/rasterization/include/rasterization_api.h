#pragma once

#include <torch/extension.h>
#include <tuple>

namespace inpc::rasterization {

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
    forward_wrapper(
        const torch::Tensor& positions,
        const torch::Tensor& features,
        const torch::Tensor& opacities,
        const torch::Tensor& w2c,
        const torch::Tensor& cam_position,
        const uint width,
        const uint height,
        const float focal_x,
        const float focal_y,
        const float principal_offset_x,
        const float principal_offset_y,
        const float near_plane,
        const float far_plane);

    std::tuple<torch::Tensor, torch::Tensor>
    backward_wrapper(
        const torch::Tensor& grad_image,
        const torch::Tensor& grad_alpha,
        const torch::Tensor& image,
        const torch::Tensor& alpha,
        const torch::Tensor& positions,
        const torch::Tensor& opacities,
        const torch::Tensor& per_point_buffers,
        const torch::Tensor& per_pixel_buffers,
        const torch::Tensor& cam_position,
        const int fragment_point_indices_selector);

    torch::Tensor
    render_default_wrapper(
        const torch::Tensor& positions,
        const torch::Tensor& features_raw,
        const torch::Tensor& bg_image,
        const torch::Tensor& w2c,
        const torch::Tensor& cam_position,
        const uint width,
        const uint height,
        const float focal_x,
        const float focal_y,
        const float principal_offset_x,
        const float principal_offset_y,
        const float near_plane,
        const float far_plane,
        const uint n_multisamples);

    torch::Tensor
    render_preextracted_wrapper(
        const torch::Tensor& positions,
        const torch::Tensor& features,
        const torch::Tensor& opacities,
        const torch::Tensor& bg_image,
        const torch::Tensor& w2c,
        const torch::Tensor& cam_position,
        const uint width,
        const uint height,
        const float focal_x,
        const float focal_y,
        const float principal_offset_x,
        const float principal_offset_y,
        const float near_plane,
        const float far_plane);

}
