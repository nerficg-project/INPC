#include "sampling_api.h"
#include "torch_utils.h"
#include "sampling.h"
#include "helper_math.h"
#include <functional>
#include <cstdint>

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

std::tuple<torch::Tensor, torch::Tensor> cuda_api::ProbabilityFieldSampler::generate_training_samples_wrapper(
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
    const torch::TensorOptions int32_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const torch::TensorOptions int64_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor sample_positions = torch::empty({n_samples, 3}, float_options);
    torch::Tensor sample_indices = torch::empty({n_samples}, int64_options);
    torch::Tensor sample_counter = torch::zeros({1}, int32_options);
    torch::Tensor viewpoint_weights = torch::empty({n_cells}, float_options);
    torch::Tensor viewpoint_weights_sum = torch::empty({1}, float_options);
    torch::Tensor sum_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> sum_buffers_func = resize_function_wrapper(sum_buffers);

    create_training_samples(
        sum_buffers_func,
        reinterpret_cast<float3*>(centers.contiguous().data_ptr<float>()),
        levels.contiguous().data_ptr<int>(),
        weights.contiguous().data_ptr<float>(),
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
        viewpoint_weights.data_ptr<float>(),
        viewpoint_weights_sum.data_ptr<float>(),
        reinterpret_cast<float3*>(sample_positions.data_ptr<float>()),
        sample_indices.data_ptr<int64_t>(),
        sample_counter.data_ptr<int>(),
        n_cells,
        n_samples,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        near_plane,
        far_plane,
        initial_size,
        this->rng);

    return {sample_positions, sample_indices};
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

    const bool valid_cdf = compute_cdf(
        sum_buffers_func,
        viewpoint_weights.data_ptr<float>(),
        viewpoint_weights_sum.data_ptr<float>(),
        n_cells);

    if (!valid_cdf) return torch::empty({0, 3}, float_options);

    torch::Tensor sample_positions = torch::empty({n_samples, 3}, float_options);
    create_samples(
        centers_ptr,
        levels_ptr,
        viewpoint_weights.data_ptr<float>(),
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

    if (n_created_samples == 0) return torch::empty({0, 3}, float_options);

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
