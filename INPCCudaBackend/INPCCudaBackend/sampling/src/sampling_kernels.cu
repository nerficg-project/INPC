#include "sampling_kernels.h"
#include "helper_math.h"
#include "pcg32.h"
#include "utils.h"
#include "sampling_utils.h"

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
        const float near,
        const float far,
        const float initial_size)
    {
        const uint cell_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (cell_idx >= n_cells) return;
        // check visibility
        const float3 center = centers[cell_idx];
        const int level = levels[cell_idx];
        const float3 p_view = transform_point_affine(w2c, center);
        const float z = -p_view.z;
        const float inv_z = 1.0f / (z + FLT_EPS); // TODO: could avoid the epsilon here
        const float x = p_view.x * inv_z * fx + cx;
        const float y = p_view.y * inv_z * fy + cy;
        bool valid = (z >= near) & (z <= far)
                & (x >= 0.0f) & (x < static_cast<float>(width - 1))
                & (y >= 0.0f) & (y < static_cast<float>(height - 1));
        if (!valid) {
            const float half_size = initial_size / float(pow(2, level + 1));
            auto check_if_valid = [&](const float3& offset) {
                const float3 corner_view = transform_point_affine(w2c, center + half_size * offset);
                const float corner_z = -corner_view.z;
                const float corner_inv_z = 1.0f / (corner_z + FLT_EPS); // TODO: could avoid the epsilon here
                const float corner_x = corner_view.x * corner_inv_z * fx + cx;
                const float corner_y = corner_view.y * corner_inv_z * fy + cy;
                return (corner_z >= near) & (corner_z <= far)
                    & (corner_x >= 0.0f) & (corner_x < static_cast<float>(width - 1))
                    & (corner_y >= 0.0f) & (corner_y < static_cast<float>(height - 1));
            };
            valid = check_if_valid(make_float3(-1.0f, -1.0f, -1.0f))
                | check_if_valid(make_float3(-1.0f, -1.0f, 1.0f))
                | check_if_valid(make_float3(-1.0f, 1.0f, -1.0f))
                | check_if_valid(make_float3(-1.0f, 1.0f, 1.0f))
                | check_if_valid(make_float3(1.0f, -1.0f, -1.0f))
                | check_if_valid(make_float3(1.0f, -1.0f, 1.0f))
                | check_if_valid(make_float3(1.0f, 1.0f, -1.0f))
                | check_if_valid(make_float3(1.0f, 1.0f, 1.0f));
        }
        // compute weight
        float viewpoint_weight = 0.0f;
        if (valid) {
            const float depth_scale = fmaxf(fabsf(z - near) / far, FLT_EPS);
            const float size_scale = exp2f(static_cast<float>(level) * 0.5f);
            viewpoint_weight = 1.0f / (depth_scale * size_scale) * weights[cell_idx];
        }
        viewpoint_weights[cell_idx] = viewpoint_weight;
    }

    __global__ void normalize_weights_cu(
        const float* viewpoint_weights_sum,
        float* viewpoint_weights,
        const uint n_weights)
    {
        const uint weight_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (weight_idx >= n_weights) return;
        const float norm_factor = 1.0f / viewpoint_weights_sum[0]; // TODO: doing the actual division could be better
        viewpoint_weights[weight_idx] *= norm_factor; // TODO: could clamp to [0, 1] to handle edge cases
    }

    __global__ void create_samples_cu(
        const float3* centers,
        const int* levels,
        const float* cdf,
        float3* sample_positions,
        const uint n_samples,
        const uint n_cells,
        const float initial_size,
        pcg32 rng)
    {
        const uint sample_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (sample_idx >= n_samples) return;
        // find cell index
        rng.advance(sample_idx * 4);
        const float sample = rng.next_float();
        const uint cell_idx = binary_search(cdf, sample, n_cells);
        // compute sample position
        const float3 random_offset = make_float3(rng.next_float(), rng.next_float(), rng.next_float()) - 0.5f;
        const float scale = 1.0f / static_cast<float>(pow(2, levels[cell_idx])) * initial_size;
        const float3 sample_position = centers[cell_idx] + random_offset * scale;
        // store sample
        sample_positions[sample_idx] = sample_position;
    }

    __global__ void compute_sample_counts_cu(
        const float* viewpoint_weights,
        const float* viewpoint_weights_sum,
        int* sample_counts,
        const uint n_samples,
        const uint n_cells)
    {
        const uint cell_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (cell_idx >= n_cells) return;
        const float viewpoint_weight = viewpoint_weights[cell_idx];
        int sample_count = 0;
        if (viewpoint_weight > 0.0f) {
            const float norm_factor = 1.0f / viewpoint_weights_sum[0]; // TODO: could do the actual division instead
            sample_count = __float2int_rn(viewpoint_weight * norm_factor * static_cast<float>(n_samples));
        }
        sample_counts[cell_idx] = sample_count;
    }

    __global__ void repeat_interleave_indices_cu(
        const int* sample_counts,
        const int* sample_offsets,
        int* sample_cell_indices,
        const uint n_cells)
    {
        const int cell_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (cell_idx >= n_cells) return;
        const int count = sample_counts[cell_idx];
        const int offset = cell_idx == 0 ? 0 : sample_offsets[cell_idx - 1];
        // TODO: load balancing -> e.g., 1 warp could do the work for 32 cells collaboratively
        for (int i = 0; i < count; ++i) sample_cell_indices[offset + i] = cell_idx;
    }

    __global__ void create_expected_samples_cu(
        const float3* centers,
        const int* levels,
        const int* sample_cell_indices,
        float3* sample_positions,
        const uint n_samples_total,
        const uint n_created_samples,
        const float initial_size,
        pcg32 rng)
    {
        const uint sample_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (sample_idx >= n_samples_total) return;
        const uint sample_idx_in_bin = sample_idx % n_created_samples;
        const uint cell_idx = sample_cell_indices[sample_idx_in_bin];
        // compute sample position
        rng.advance(sample_idx * 3);
        const float3 random_offset = make_float3(rng.next_float(), rng.next_float(), rng.next_float()) - 0.5f;
        const float scale = 1.0f / static_cast<float>(pow(2, levels[cell_idx])) * initial_size;
        const float3 sample_position = centers[cell_idx] + random_offset * scale;
        sample_positions[sample_idx] = sample_position;
    }

}
