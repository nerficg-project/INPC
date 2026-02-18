#pragma once

#include "helper_math.h"
#include "utils.h"
#include <functional>
#include <cub/cub.cuh>

namespace inpc::sampling {

    template <bool workspace_only>
    struct SumBuffers {
        size_t cub_workspace_size;
        char* cub_workspace;
        int* sample_counts = nullptr;
        int* sample_offsets = nullptr;

        static SumBuffers from_blob(char*& blob, const uint n_cells) {
            SumBuffers buffers;
            float* float_ptr = nullptr;
            size_t reduce_sum_size = 0;
            cub::DeviceReduce::Sum(
                nullptr, reduce_sum_size,
                float_ptr, float_ptr,
                n_cells
            );
            size_t prefix_sum_size = 0;
            cub::DeviceScan::InclusiveSum(
                nullptr, prefix_sum_size,
                float_ptr, float_ptr,
                n_cells
            );
            buffers.cub_workspace_size = max(reduce_sum_size, prefix_sum_size);
            if constexpr (!workspace_only) {
                obtain(blob, buffers.sample_counts, n_cells, 128);
                obtain(blob, buffers.sample_offsets, n_cells, 128);
                size_t int_prefix_sum_size = 0;
                cub::DeviceScan::InclusiveSum(
                    nullptr, int_prefix_sum_size,
                    buffers.sample_counts, buffers.sample_offsets,
                    n_cells
                );
                buffers.cub_workspace_size = max(buffers.cub_workspace_size, int_prefix_sum_size);
            }
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
            return buffers;
        }
    };

    __device__ inline uint binary_search(
        const float* cdf,
        const float& sample,
        const uint& n)
    {
        uint it;
        uint count, step;
        count = n;
        uint first = 0;
        while (count > 0) {
            it = first;
            step = count / 2;
            it += step;
            if (cdf[it] < sample) {
                first = ++it;
                count -= step + 1;
            }
            else count = step;
        }
        return min(first, n - 1);
    }

}
