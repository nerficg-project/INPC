#include "rasterization_api.h"
#include "torch_utils.h"
#include "rasterizer.h"
#include "helper_math.h"
#include "fp16_utils.h"

namespace cuda_api = inpc::rasterization;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
cuda_api::forward_wrapper(
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
    const float far_plane)
{
    const uint n_points = positions.size(0);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor image = torch::empty({4, height, width}, float_options);
    torch::Tensor alpha = torch::empty({1, height, width}, float_options);
    torch::Tensor blending_weights = torch::zeros({n_points}, float_options);
    torch::Tensor per_point_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_pixel_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> per_point_buffers_func = resize_function_wrapper(per_point_buffers);
    const std::function<char*(size_t)> per_pixel_buffers_func = resize_function_wrapper(per_pixel_buffers);

    const int fragment_point_indices_selector = training::forward(
        per_point_buffers_func,
        per_pixel_buffers_func,
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(positions.contiguous().data_ptr<float>()),
        reinterpret_cast<half4*>(features.contiguous().data_ptr<at::Half>()),
        opacities.contiguous().data_ptr<float>(),
        image.contiguous().data_ptr<float>(),
        alpha.contiguous().data_ptr<float>(),
        blending_weights.contiguous().data_ptr<float>(),
        n_points,
        width,
        height,
        focal_x,
        focal_y,
        principal_offset_x + static_cast<float>(width) * 0.5f,
        principal_offset_y + static_cast<float>(height) * 0.5f,
        near_plane,
        far_plane);

    return std::make_tuple(image, alpha, blending_weights, per_point_buffers, per_pixel_buffers, fragment_point_indices_selector);
}

std::tuple<torch::Tensor, torch::Tensor>
cuda_api::backward_wrapper(
    const torch::Tensor& grad_image,
    const torch::Tensor& grad_alpha,
    const torch::Tensor& image,
    const torch::Tensor& alpha,
    const torch::Tensor& positions,
    const torch::Tensor& opacities,
    const torch::Tensor& per_point_buffers,
    const torch::Tensor& per_pixel_buffers,
    const torch::Tensor& cam_position,
    const int fragment_point_indices_selector)
{
    const uint n_points = positions.size(0);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions half_options = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);
    torch::Tensor grad_features_view = torch::zeros({n_points, 4}, float_options);
    torch::Tensor grad_features = torch::zeros({n_points, 36}, half_options);
    torch::Tensor grad_opacities = torch::zeros({n_points, 1}, float_options);

    training::backward(
        grad_image.contiguous().data_ptr<float>(),
        grad_alpha.contiguous().data_ptr<float>(),
        image.contiguous().data_ptr<float>(),
        alpha.contiguous().data_ptr<float>(),
        reinterpret_cast<float3*>(positions.contiguous().data_ptr<float>()),
        opacities.contiguous().data_ptr<float>(),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<char*>(per_point_buffers.contiguous().data_ptr()),
        reinterpret_cast<char*>(per_pixel_buffers.contiguous().data_ptr()),
        reinterpret_cast<float4*>(grad_features_view.contiguous().data_ptr<float>()),
        reinterpret_cast<half4*>(grad_features.contiguous().data_ptr<at::Half>()),
        grad_opacities.contiguous().data_ptr<float>(),
        n_points,
        image.size(1) * image.size(2),
        image.size(2),
        fragment_point_indices_selector);

    return std::make_tuple(grad_features, grad_opacities);
}

torch::Tensor
cuda_api::render_default_wrapper(
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
    const uint n_multisamples)
{
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor image = torch::zeros({4, height, width}, float_options);
    torch::Tensor per_point_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_pixel_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> per_point_buffers_func = resize_function_wrapper(per_point_buffers);
    const std::function<char*(size_t)> per_pixel_buffers_func = resize_function_wrapper(per_pixel_buffers);

    inference::render(
        per_point_buffers_func,
        per_pixel_buffers_func,
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(positions.contiguous().data_ptr<float>()),
        reinterpret_cast<__half*>(features_raw.contiguous().data_ptr<at::Half>()),
        reinterpret_cast<half4*>(bg_image.contiguous().data_ptr<at::Half>()),
        image.contiguous().data_ptr<float>(),
        positions.size(0),
        width,
        height,
        focal_x,
        focal_y,
        principal_offset_x + static_cast<float>(width) * 0.5f,
        principal_offset_y + static_cast<float>(height) * 0.5f,
        near_plane,
        far_plane,
        n_multisamples);

    return image;
}

torch::Tensor
cuda_api::render_preextracted_wrapper(
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
    const float far_plane)
{
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor image = torch::empty({4, height, width}, float_options);
    torch::Tensor per_point_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_fragment_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_pixel_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> per_point_buffers_func = resize_function_wrapper(per_point_buffers);
    const std::function<char*(size_t)> per_fragment_buffers_func = resize_function_wrapper(per_fragment_buffers);
    const std::function<char*(size_t)> per_pixel_buffers_func = resize_function_wrapper(per_pixel_buffers);

    inference_preextracted::render(
        per_point_buffers_func,
        per_fragment_buffers_func,
        per_pixel_buffers_func,
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(positions.contiguous().data_ptr<float>()),
        reinterpret_cast<half4*>(features.contiguous().data_ptr<at::Half>()),
        opacities.contiguous().data_ptr<float>(),
        reinterpret_cast<half4*>(bg_image.contiguous().data_ptr<at::Half>()),
        image.contiguous().data_ptr<float>(),
        positions.size(0),
        width,
        height,
        focal_x,
        focal_y,
        principal_offset_x + static_cast<float>(width) * 0.5f,
        principal_offset_y + static_cast<float>(height) * 0.5f,
        near_plane,
        far_plane);

    return image;
}
