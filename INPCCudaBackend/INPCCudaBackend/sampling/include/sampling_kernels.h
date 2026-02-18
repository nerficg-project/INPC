#pragma once

#include "pcg32.h"
#include "helper_math.h"

namespace inpc::sampling {

    __global__ void compute_viewpoint_weights_cu(
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
    
    __global__ void normalize_weights_cu(
        const float* viewpoint_weights_sum,
        float* viewpoint_weights,
        const uint n_weights);
    
    __global__ void create_samples_cu(
        const float3* centers,
        const int* levels,
        const float* cdf,
        float3* sample_positions,
        const uint n_samples,
        const uint n_cells,
        const float initial_size,
        pcg32 rng);

    __global__ void compute_sample_counts_cu(
        const float* viewpoint_weights,
        const float* viewpoint_weights_sum,
        int* sample_counts,
        const uint n_samples,
        const uint n_cells);

    __global__ void repeat_interleave_indices_cu(
        const int* sample_counts,
        const int* sample_offsets,
        int* sample_cell_indices,
        const uint n_cells);

    __global__ void create_expected_samples_cu(
        const float3* centers,
        const int* levels,
        const int* sample_cell_indices,
        float3* sample_positions,
        const uint n_samples_total,
        const uint n_created_samples,
        const float initial_size,
        pcg32 rng);

}
