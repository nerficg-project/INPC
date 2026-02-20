#include "rasterization_api.h"

#include "bilinear/forward.h"
#include "bilinear/backward.h"
#include "bilinear/inference.h"
#include "bilinear/inference_preextracted.h"

#include "gaussian/forward.h"
#include "gaussian/backward.h"
#include "gaussian/inference.h"
#include "gaussian/inference_preextracted.h"

#include "torch_utils.h"
#include "helper_math.h"
#include <stdexcept>
#include <functional>

enum class RasterizerMode {
    BILINEAR = 0,
    GAUSSIAN = 1,
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int>
inpc::rasterization::forward_wrapper(
    const torch::Tensor& positions,
    const torch::Tensor& features,
    const torch::Tensor& opacities,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    const int rasterizer_mode,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane)
{
    const RasterizerMode mode = static_cast<RasterizerMode>(rasterizer_mode);
    const int n_primitives = positions.size(0);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor image = torch::empty({4, height, width}, float_options);
    torch::Tensor alpha = torch::empty({1, height, width}, float_options);
    torch::Tensor blending_weights = torch::zeros({n_primitives}, float_options);
    torch::Tensor per_primitive_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_tile_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_instance_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> per_primitive_buffers_func = resize_function_wrapper(per_primitive_buffers);
    const std::function<char*(size_t)> per_tile_buffers_func = resize_function_wrapper(per_tile_buffers);
    const std::function<char*(size_t)> per_instance_buffers_func = resize_function_wrapper(per_instance_buffers);

    std::pair<int, int> buffer_state;
    switch (mode) {
        case RasterizerMode::BILINEAR:
            buffer_state = bilinear::forward(
                per_primitive_buffers_func,
                per_tile_buffers_func,
                per_instance_buffers_func,
                reinterpret_cast<float3*>(positions.contiguous().data_ptr<float>()),
                reinterpret_cast<half4*>(features.contiguous().data_ptr<at::Half>()),
                opacities.contiguous().data_ptr<float>(),
                reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
                reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
                image.data_ptr<float>(),
                alpha.data_ptr<float>(),
                blending_weights.data_ptr<float>(),
                n_primitives,
                width,
                height,
                focal_x,
                focal_y,
                center_x,
                center_y,
                near_plane,
                far_plane);
            break;
        case RasterizerMode::GAUSSIAN:
            buffer_state = gaussian::forward(
                per_primitive_buffers_func,
                per_tile_buffers_func,
                per_instance_buffers_func,
                reinterpret_cast<float3*>(positions.contiguous().data_ptr<float>()),
                reinterpret_cast<half4*>(features.contiguous().data_ptr<at::Half>()),
                opacities.contiguous().data_ptr<float>(),
                reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
                reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
                image.data_ptr<float>(),
                alpha.data_ptr<float>(),
                blending_weights.data_ptr<float>(),
                n_primitives,
                width,
                height,
                focal_x,
                focal_y,
                center_x,
                center_y,
                near_plane,
                far_plane);
            break;
        default:
            throw std::runtime_error("unsupported rasterizer mode");
    }

    return {image, alpha, blending_weights, per_primitive_buffers, per_tile_buffers, per_instance_buffers, buffer_state.first, buffer_state.second};
}

std::tuple<torch::Tensor, torch::Tensor>
inpc::rasterization::backward_wrapper(
    const torch::Tensor& grad_image,
    const torch::Tensor& grad_alpha,
    const torch::Tensor& image,
    const torch::Tensor& alpha,
    const torch::Tensor& positions,
    const torch::Tensor& opacities,
    const torch::Tensor& per_primitive_buffers,
    const torch::Tensor& per_tile_buffers,
    const torch::Tensor& per_instance_buffers,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    const int rasterizer_mode,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const int n_instances,
    const int instance_primitive_indices_selector)
{
    const RasterizerMode mode = static_cast<RasterizerMode>(rasterizer_mode);
    const int n_primitives = positions.size(0);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions half_options = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);
    torch::Tensor grad_features_view = torch::zeros({n_primitives, 4}, float_options);
    torch::Tensor grad_features = torch::zeros({n_primitives, 36}, half_options);
    torch::Tensor grad_opacities = torch::zeros({n_primitives, 1}, float_options);

    switch (mode) {
        case RasterizerMode::BILINEAR:
            bilinear::backward(
                grad_image.contiguous().data_ptr<float>(),
                grad_alpha.contiguous().data_ptr<float>(),
                image.contiguous().data_ptr<float>(),
                alpha.contiguous().data_ptr<float>(),
                reinterpret_cast<float3*>(positions.contiguous().data_ptr<float>()),
                opacities.contiguous().data_ptr<float>(),
                reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
                reinterpret_cast<char*>(per_primitive_buffers.data_ptr()),
                reinterpret_cast<char*>(per_tile_buffers.data_ptr()),
                reinterpret_cast<char*>(per_instance_buffers.data_ptr()),
                reinterpret_cast<float4*>(grad_features_view.data_ptr<float>()),
                reinterpret_cast<half4*>(grad_features.data_ptr<at::Half>()),
                grad_opacities.data_ptr<float>(),
                n_primitives,
                width,
                height,
                n_instances,
                instance_primitive_indices_selector);
            break;
        case RasterizerMode::GAUSSIAN:
            gaussian::backward(
                grad_image.contiguous().data_ptr<float>(),
                grad_alpha.contiguous().data_ptr<float>(),
                image.contiguous().data_ptr<float>(),
                alpha.contiguous().data_ptr<float>(),
                reinterpret_cast<float3*>(positions.contiguous().data_ptr<float>()),
                opacities.contiguous().data_ptr<float>(),
                reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
                reinterpret_cast<char*>(per_primitive_buffers.data_ptr()),
                reinterpret_cast<char*>(per_tile_buffers.data_ptr()),
                reinterpret_cast<char*>(per_instance_buffers.data_ptr()),
                reinterpret_cast<float4*>(grad_features_view.data_ptr<float>()),
                reinterpret_cast<half4*>(grad_features.data_ptr<at::Half>()),
                grad_opacities.data_ptr<float>(),
                n_primitives,
                width,
                height,
                n_instances,
                instance_primitive_indices_selector);
            break;
        default:
            throw std::runtime_error("unsupported rasterizer mode");
    }

    return {grad_features, grad_opacities};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
inpc::rasterization::inference_wrapper(
    const torch::Tensor& positions,
    const torch::Tensor& features_raw,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    const int rasterizer_mode,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const float sigma_world,
    const float sigma_cutoff)
{
    const RasterizerMode mode = static_cast<RasterizerMode>(rasterizer_mode);
    const int n_primitives = positions.size(0);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor image = torch::empty({4, height, width}, float_options);
    torch::Tensor depth = torch::empty({1, height, width}, float_options);
    torch::Tensor alpha = torch::empty({1, height, width}, float_options);
    torch::Tensor per_primitive_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_tile_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_instance_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> per_primitive_buffers_func = resize_function_wrapper(per_primitive_buffers);
    const std::function<char*(size_t)> per_tile_buffers_func = resize_function_wrapper(per_tile_buffers);
    const std::function<char*(size_t)> per_instance_buffers_func = resize_function_wrapper(per_instance_buffers);

    switch (mode) {
        case RasterizerMode::BILINEAR:
            bilinear::inference(
                per_primitive_buffers_func,
                per_tile_buffers_func,
                per_instance_buffers_func,
                reinterpret_cast<float3*>(positions.contiguous().data_ptr<float>()),
                reinterpret_cast<__half*>(features_raw.contiguous().data_ptr<at::Half>()),
                reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
                reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
                image.data_ptr<float>(),
                depth.data_ptr<float>(),
                alpha.data_ptr<float>(),
                n_primitives,
                width,
                height,
                focal_x,
                focal_y,
                center_x,
                center_y,
                near_plane,
                far_plane);
            break;
        case RasterizerMode::GAUSSIAN:
            gaussian::inference(
                per_primitive_buffers_func,
                per_tile_buffers_func,
                per_instance_buffers_func,
                reinterpret_cast<float3*>(positions.contiguous().data_ptr<float>()),
                reinterpret_cast<__half*>(features_raw.contiguous().data_ptr<at::Half>()),
                reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
                reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
                image.data_ptr<float>(),
                depth.data_ptr<float>(),
                alpha.data_ptr<float>(),
                n_primitives,
                width,
                height,
                focal_x,
                focal_y,
                center_x,
                center_y,
                near_plane,
                far_plane,
                sigma_world,
                sigma_cutoff);
            break;
        default:
            throw std::runtime_error("unsupported rasterizer mode");
    }
    
    return {image, depth, alpha};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
inpc::rasterization::inference_preextracted_wrapper(
    const torch::Tensor& positions,
    const torch::Tensor& features,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    const int rasterizer_mode,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const float sigma_world,
    const float sigma_cutoff)
{
    const RasterizerMode mode = static_cast<RasterizerMode>(rasterizer_mode);
    const int n_primitives = positions.size(0);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor image = torch::empty({4, height, width}, float_options);
    torch::Tensor depth = torch::empty({1, height, width}, float_options);
    torch::Tensor alpha = torch::empty({1, height, width}, float_options);
    torch::Tensor per_primitive_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_tile_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_instance_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> per_primitive_buffers_func = resize_function_wrapper(per_primitive_buffers);
    const std::function<char*(size_t)> per_tile_buffers_func = resize_function_wrapper(per_tile_buffers);
    const std::function<char*(size_t)> per_instance_buffers_func = resize_function_wrapper(per_instance_buffers);

    switch (mode) {
        case RasterizerMode::BILINEAR:
            bilinear::inference_preextracted(
                per_primitive_buffers_func,
                per_tile_buffers_func,
                per_instance_buffers_func,
                reinterpret_cast<float3*>(positions.contiguous().data_ptr<float>()),
                reinterpret_cast<half4*>(features.contiguous().data_ptr<at::Half>()),
                opacities.contiguous().data_ptr<float>(),
                reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
                reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
                image.data_ptr<float>(),
                depth.data_ptr<float>(),
                alpha.data_ptr<float>(),
                n_primitives,
                width,
                height,
                focal_x,
                focal_y,
                center_x,
                center_y,
                near_plane,
                far_plane);
            break;
        case RasterizerMode::GAUSSIAN:
            gaussian::inference_preextracted(
                per_primitive_buffers_func,
                per_tile_buffers_func,
                per_instance_buffers_func,
                reinterpret_cast<float3*>(positions.contiguous().data_ptr<float>()),
                reinterpret_cast<half4*>(features.contiguous().data_ptr<at::Half>()),
                opacities.contiguous().data_ptr<float>(),
                scales.size(0) != 0 ? scales.contiguous().data_ptr<float>() : nullptr,
                reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
                reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
                image.data_ptr<float>(),
                depth.data_ptr<float>(),
                alpha.data_ptr<float>(),
                n_primitives,
                width,
                height,
                focal_x,
                focal_y,
                center_x,
                center_y,
                near_plane,
                far_plane,
                sigma_world,
                sigma_cutoff);
            break;
        default:
            throw std::runtime_error("unsupported rasterizer mode");
    }
    
    return {image, depth, alpha};
}
