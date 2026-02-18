#include "sampling_api.h"
#include "torch_utils.h"
#include "sampling.h"
#include "helper_math.h"
#include <functional>

namespace cuda_api = inpc::sampling;

torch::Tensor cuda_api::compute_viewpoint_weights_wrapper(
    const torch::Tensor& centers,
    const torch::Tensor& levels,
    const torch::Tensor& weights,
    const torch::Tensor& w2c,
    const uint width,
    const uint height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const float initial_size)
{
    const uint n_cells = centers.size(0);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    torch::Tensor viewpoint_weights = torch::empty({n_cells}, float_options);

    compute_viewpoint_weights(
        reinterpret_cast<const float3*>(centers.contiguous().data_ptr<float>()),
        levels.contiguous().data_ptr<int>(),
        weights.contiguous().data_ptr<float>(),
        reinterpret_cast<const float4*>(w2c.contiguous().data_ptr<float>()),
        viewpoint_weights.data_ptr<float>(),
        n_cells,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        near_plane,
        far_plane,
        initial_size);

    return viewpoint_weights;
}

torch::Tensor cuda_api::ProbabilityFieldSampler::generate_samples_wrapper(
    const torch::Tensor& centers,
    const torch::Tensor& levels,
    const torch::Tensor& weights,
    const torch::Tensor& w2c,
    const uint n_samples,
    const uint width,
    const uint height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const float initial_size)
{
    const uint n_cells = centers.size(0);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor sample_positions = torch::empty({n_samples, 3}, float_options);
    torch::Tensor viewpoint_weights = torch::empty({n_cells}, float_options);
    torch::Tensor viewpoint_weights_sum = torch::empty({1}, float_options);
    torch::Tensor sum_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> sum_buffers_func = resize_function_wrapper(sum_buffers);
    const float3* centers_ptr = reinterpret_cast<const float3*>(centers.contiguous().data_ptr<float>());
    const int* levels_ptr = levels.contiguous().data_ptr<int>();

    compute_viewpoint_weights(
        centers_ptr,
        levels_ptr,
        weights.contiguous().data_ptr<float>(),
        reinterpret_cast<const float4*>(w2c.contiguous().data_ptr<float>()),
        viewpoint_weights.data_ptr<float>(),
        n_cells,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        near_plane,
        far_plane,
        initial_size);

    create_samples(
        sum_buffers_func,
        centers_ptr,
        levels_ptr,
        viewpoint_weights.data_ptr<float>(),
        viewpoint_weights_sum.data_ptr<float>(),
        reinterpret_cast<float3*>(sample_positions.data_ptr<float>()),
        n_samples,
        n_cells,
        initial_size,
        this->rng);

    return sample_positions;
}

torch::Tensor cuda_api::ProbabilityFieldSampler::generate_expected_samples_wrapper(
    const torch::Tensor& centers,
    const torch::Tensor& levels,
    const torch::Tensor& weights,
    const torch::Tensor& w2c,
    const uint n_samples,
    const uint n_multi,
    const uint width,
    const uint height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const float initial_size)
{
    const uint n_cells = centers.size(0);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions int_options = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor viewpoint_weights = torch::empty({n_cells}, float_options);
    torch::Tensor viewpoint_weights_sum = torch::empty({1}, float_options);
    torch::Tensor sum_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> sum_buffers_func = resize_function_wrapper(sum_buffers);
    const float3* centers_ptr = reinterpret_cast<const float3*>(centers.contiguous().data_ptr<float>());
    const int* levels_ptr = levels.contiguous().data_ptr<int>();

    compute_viewpoint_weights(
        centers_ptr,
        levels_ptr,
        weights.contiguous().data_ptr<float>(),
        reinterpret_cast<const float4*>(w2c.contiguous().data_ptr<float>()),
        viewpoint_weights.data_ptr<float>(),
        n_cells,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        near_plane,
        far_plane,
        initial_size);

    const uint n_created_samples = compute_sample_distribution(
        sum_buffers_func,
        viewpoint_weights.data_ptr<float>(),
        viewpoint_weights_sum.data_ptr<float>(),
        n_samples,
        n_cells);
    
    torch::Tensor sample_cell_indices = torch::empty({n_created_samples}, int_options);
    const uint n_samples_total = n_created_samples * n_multi;
    torch::Tensor sample_positions = torch::empty({n_samples_total, 3}, float_options);
    create_expected_samples(
        centers_ptr,
        levels_ptr,
        reinterpret_cast<char*>(sum_buffers.contiguous().data_ptr()),
        sample_cell_indices.data_ptr<int>(),
        reinterpret_cast<float3*>(sample_positions.data_ptr<float>()),
        n_samples_total,
        n_created_samples,
        n_cells,
        initial_size,
        this->rng);

    return sample_positions;
}
