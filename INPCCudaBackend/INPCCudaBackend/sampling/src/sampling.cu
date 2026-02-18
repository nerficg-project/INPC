#include "sampling.h"
#include "sampling_config.h"
#include "sampling_utils.h"
#include "sampling_kernels.h"
#include "helper_math.h"
#include "utils.h"
#include "pcg32.h"
#include <functional>
#include <cub/cub.cuh>

namespace cuda_api = inpc::sampling;

void cuda_api::compute_viewpoint_weights(
    const float3* centers,
    const int* levels,
    const float* weights,
    const float4* w2c,
    float* viewpoint_weights,
    const uint n_cells,
    const uint width,
    const uint height,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const float near_plane,
    const float far_plane,
    const float initial_size)
{
    compute_viewpoint_weights_cu<<<div_round_up(n_cells, config::block_size_compute_weights), config::block_size_compute_weights>>>(
        centers,
        levels,
        weights,
        w2c,
        viewpoint_weights,
        n_cells,
        width,
        height,
        fx,
        fy,
        cx,
        cy,
        near_plane,
        far_plane,
        initial_size);
    CHECK_CUDA(config::debug, "compute_viewpoint_weights_cu");
}

void cuda_api::create_samples(
    std::function<char* (size_t)> sum_buffers_func,
    const float3* centers,
    const int* levels,
    float* viewpoint_weights,
    float* viewpoint_weights_sum,
    float3* sample_positions,
    const uint n_samples,
    const uint n_cells,
    const float initial_size,
    pcg32& rng)
 {
    char* sum_buffers_blob = sum_buffers_func(required<SumBuffers<true>>(n_cells));
    SumBuffers sum_buffers = SumBuffers<true>::from_blob(sum_buffers_blob, n_cells);

    cub::DeviceReduce::Sum(
        sum_buffers.cub_workspace,
        sum_buffers.cub_workspace_size,
        viewpoint_weights,
        viewpoint_weights_sum,
        n_cells);
    CHECK_CUDA(config::debug, "cub::DeviceReduce::Sum");

    // TODO: doing the inclusive sum first might be more numerically stable
    normalize_weights_cu<<<div_round_up(n_cells, config::block_size_normalize_weights), config::block_size_normalize_weights>>>(
        viewpoint_weights_sum,
        viewpoint_weights,
        n_cells);
    CHECK_CUDA(config::debug, "normalize_weights_cu");

    cub::DeviceScan::InclusiveSum(
        sum_buffers.cub_workspace,
        sum_buffers.cub_workspace_size,
        viewpoint_weights,
        viewpoint_weights,
        n_cells);
    CHECK_CUDA(config::debug, "cub::DeviceScan::InclusiveSum");

    create_samples_cu<<<div_round_up(n_samples, config::block_size_create_samples), config::block_size_create_samples>>>(
        centers,
        levels,
        viewpoint_weights,
        sample_positions,
        n_samples,
        n_cells,
        initial_size,
        rng);
    CHECK_CUDA(config::debug, "create_samples_cu");
    rng.advance(n_samples * 4);
}

uint cuda_api::compute_sample_distribution(
    std::function<char* (size_t)> sum_buffers_func,
    float* viewpoint_weights,
    float* viewpoint_weights_sum,
    const uint n_samples,
    const uint n_cells)
{
    char* sum_buffers_blob = sum_buffers_func(required<SumBuffers<false>>(n_cells));
    SumBuffers sum_buffers = SumBuffers<false>::from_blob(sum_buffers_blob, n_cells);

    cub::DeviceReduce::Sum(
        sum_buffers.cub_workspace,
        sum_buffers.cub_workspace_size,
        viewpoint_weights,
        viewpoint_weights_sum,
        n_cells);
    CHECK_CUDA(config::debug, "cub::DeviceReduce::Sum");

    compute_sample_counts_cu<<<div_round_up(n_cells, config::block_size_compute_sample_counts), config::block_size_compute_sample_counts>>>(
        viewpoint_weights,
        viewpoint_weights_sum,
        sum_buffers.sample_counts,
        n_samples,
        n_cells);
    CHECK_CUDA(config::debug, "compute_sample_counts_cu")

    cub::DeviceScan::InclusiveSum(
        sum_buffers.cub_workspace,
        sum_buffers.cub_workspace_size,
        sum_buffers.sample_counts,
        sum_buffers.sample_offsets,
        n_cells);
    CHECK_CUDA(config::debug, "cub::DeviceScan::InclusiveSum")

    int n_created_samples;
    cudaMemcpy(&n_created_samples, sum_buffers.sample_offsets + n_cells - 1, sizeof(int), cudaMemcpyDeviceToHost);

    return n_created_samples;
}

void cuda_api::create_expected_samples(
    const float3* centers,
    const int* levels,
    char* sum_buffers_blob,
    int* sample_cell_indices,
    float3* sample_positions,
    const uint n_samples_total,
    const uint n_created_samples,
    const uint n_cells,
    const float initial_size,
    pcg32& rng)
{
    SumBuffers sum_buffers = SumBuffers<false>::from_blob(sum_buffers_blob, n_cells);

    repeat_interleave_indices_cu<<<div_round_up(n_cells, config::block_size_repeat_interleave_indices), config::block_size_repeat_interleave_indices>>>(
        sum_buffers.sample_counts,
        sum_buffers.sample_offsets,
        sample_cell_indices,
        n_cells);
    CHECK_CUDA(config::debug, "repeat_interleave_indices_cu");

    create_expected_samples_cu<<<div_round_up(n_samples_total, config::block_size_create_expected_samples), config::block_size_create_expected_samples>>>(
        centers,
        levels,
        sample_cell_indices,
        sample_positions,
        n_samples_total,
        n_created_samples,
        initial_size,
        rng);
    CHECK_CUDA(config::debug, "create_expected_samples_cu");
    rng.advance(n_samples_total * 3);
}
