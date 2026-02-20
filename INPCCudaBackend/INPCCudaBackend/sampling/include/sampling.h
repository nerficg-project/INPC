#pragma once

#include "pcg32.h"
#include "helper_math.h"
#include <functional>

namespace inpc::sampling {

    void compute_viewpoint_weights(
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
        const float initial_size);

    void create_training_samples(
        std::function<char* (size_t)> sum_buffers_func,
        const float3* centers,
        const int* levels,
        const float* weights,
        const float4* w2c,
        float* viewpoint_weights,
        float* viewpoint_weights_sum,
        float3* sample_positions,
        int64_t* sample_indices,
        int* sample_counter,
        const uint n_cells,
        const uint n_samples,
        const uint width,
        const uint height,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float near_plane,
        const float far_plane,
        const float initial_size,
        pcg32& rng);

    bool compute_cdf(
        std::function<char* (size_t)> sum_buffers_func,
        float* viewpoint_weights,
        float* viewpoint_weights_sum,
        const uint n_cells);

    void create_samples(
        const float3* centers,
        const int* levels,
        const float* cdf,
        float3* sample_positions,
        const uint n_samples,
        const uint n_cells,
        const float initial_size,
        pcg32& rng);

    uint compute_sample_distribution(
        std::function<char* (size_t)> sum_buffers_func,
        float* viewpoint_weights,
        float* viewpoint_weights_sum,
        const uint n_samples,
        const uint n_cells);

    void create_expected_samples(
        const float3* centers,
        const int* levels,
        char* sum_buffers_blob,
        int* sample_cell_indices,
        float3* sample_positions,
        const uint n_samples_total,
        const uint n_created_samples,
        const uint n_cells,
        const float initial_size,
        pcg32& rng);

}
