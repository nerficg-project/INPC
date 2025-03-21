#pragma once

#include "helper_math.h"
#include "fp16_utils.h"
#include "utils.h"
#include <cub/cub.cuh>
#include <cstdint>
#include <functional>

namespace inpc::rasterization {

    namespace training {

        int forward(
            std::function<char* (size_t)> per_point_buffers_func,
            std::function<char* (size_t)> per_pixel_buffers_func,
            const float4* w2c,
            const float3* cam_position,
            const float3* positions,
            const half4* features,
            const float* opacities,
            float* image,
            float* alpha,
            float* blending_weights,
            const uint n_points,
            const uint width,
            const uint height,
            const float fx,
            const float fy,
            const float cx,
            const float cy,
            const float near,
            const float far);

        void backward(
            const float* grad_image,
            const float* grad_alpha,
            const float* image,
            const float* alpha,
            const float3* positions,
            const float* opacities,
            const float3* cam_position,
            char* per_point_buffers_blob,
            char* per_pixel_buffers_blob,
            float4* grad_features_view,
            half4* grad_features,
            float* grad_opacities,
            const uint n_points,
            const uint n_pixels,
            const uint width,
            const int fragment_point_indices_selector);
        
        struct PerPointBuffers {
            size_t cub_workspace_size;
            char* cub_workspace;
            float2* screen_coords;
            float4* features_view;
            cub::DoubleBuffer<uint64_t> fragment_keys;
            cub::DoubleBuffer<uint> fragment_point_indices;
    
            static PerPointBuffers from_blob(char*& blob, const size_t n_points) {
                PerPointBuffers buffers;
                obtain(blob, buffers.screen_coords, n_points, 128);
                obtain(blob, buffers.features_view, n_points, 128);
                const size_t n_fragments = n_points * 4;
                uint64_t* fragment_keys_unsorted;
                uint64_t* fragment_keys_sorted;
                uint* fragment_point_indices_unsorted;
                uint* fragment_point_indices_sorted;
                obtain(blob, fragment_keys_unsorted, n_fragments, 128);
                obtain(blob, fragment_keys_sorted, n_fragments, 128);
                obtain(blob, fragment_point_indices_unsorted, n_fragments, 128);
                obtain(blob, fragment_point_indices_sorted, n_fragments, 128);
                // using cub::DoubleBuffer halves the number of auxiliary memory required but requires passing state to the backward pass
                buffers.fragment_keys = cub::DoubleBuffer(fragment_keys_unsorted, fragment_keys_sorted);
                buffers.fragment_point_indices = cub::DoubleBuffer(fragment_point_indices_unsorted, fragment_point_indices_sorted);
                cub::DeviceRadixSort::SortPairs(
                    nullptr, buffers.cub_workspace_size,
                    buffers.fragment_keys,
                    buffers.fragment_point_indices,
                    n_fragments
                );
                obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
                return buffers;
            }
        };

        struct PerPixelBuffers {
            size_t cub_workspace_size;
            char* cub_workspace;
            uint* fragment_counts;
            uint* fragment_offsets;
    
            static PerPixelBuffers from_blob(char*& blob, const size_t n_pixels) {
                PerPixelBuffers buffers;
                obtain(blob, buffers.fragment_counts, n_pixels, 128);
                obtain(blob, buffers.fragment_offsets, n_pixels, 128);
                cub::DeviceScan::InclusiveSum(
                    nullptr, buffers.cub_workspace_size,
                    buffers.fragment_counts, buffers.fragment_offsets,
                    n_pixels
                );
                obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
                return buffers;
            }
        };

    }

    namespace inference {

        void render(
            std::function<char* (size_t)> per_point_buffers_func,
            std::function<char* (size_t)> per_pixel_buffers_func,
            const float4* w2c,
            const float3* cam_position,
            const float3* positions,
            const __half* features_raw,
            const half4* bg_image,
            float* image,
            const uint n_points,
            const uint width,
            const uint height,
            const float fx,
            const float fy,
            const float cx,
            const float cy,
            const float near,
            const float far,
            const uint n_multisamples);
        
        struct PerPointBuffers: public inpc::rasterization::training::PerPointBuffers {
            float* opacities;
    
            static PerPointBuffers from_blob(char*& blob, const size_t n_points) {
                PerPointBuffers buffers;
                inpc::rasterization::training::PerPointBuffers& base_buffers = buffers;
                base_buffers = inpc::rasterization::training::PerPointBuffers::from_blob(blob, n_points);
                obtain(blob, buffers.opacities, n_points, 128);
                return buffers;
            }
        };

    }

    namespace inference_preextracted {

        void render(
            std::function<char* (size_t)> per_point_buffers_func,
            std::function<char* (size_t)> per_fragment_buffers_func,
            std::function<char* (size_t)> per_pixel_buffers_func,
            const float4* w2c,
            const float3* cam_position,
            const float3* positions,
            const half4* features,
            const float* opacities,
            const half4* bg_image,
            float* image,
            const uint n_points,
            const uint width,
            const uint height,
            const float fx,
            const float fy,
            const float cx,
            const float cy,
            const float near,
            const float far);
        
        struct PerPointBuffers {
            size_t cub_workspace_size;
            char* cub_workspace;
            float* depths;
            float2* screen_coords;
            uint* n_visible_fragments;
            uint* offsets;
    
            static PerPointBuffers from_blob(char*& blob, const size_t n_points) {
                PerPointBuffers buffers;
                obtain(blob, buffers.n_visible_fragments, n_points, 128);
                obtain(blob, buffers.offsets, n_points, 128);
                obtain(blob, buffers.depths, n_points, 128);
                obtain(blob, buffers.screen_coords, n_points, 128);
                cub::DeviceScan::InclusiveSum(
                    nullptr, buffers.cub_workspace_size,
                    buffers.n_visible_fragments, buffers.offsets,
                    n_points
                );
                obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
                return buffers;
            }
        };
        
        struct PerFragmentBuffers {
            size_t cub_workspace_size;
            char* cub_workspace;
            cub::DoubleBuffer<uint64_t> fragment_keys;
            cub::DoubleBuffer<uint> fragment_point_indices;
    
            static PerFragmentBuffers from_blob(char*& blob, const size_t n_fragments) {
                PerFragmentBuffers buffers;
                uint64_t* fragment_keys_unsorted;
                uint64_t* fragment_keys_sorted;
                uint* fragment_point_indices_unsorted;
                uint* fragment_point_indices_sorted;
                obtain(blob, fragment_keys_unsorted, n_fragments, 128);
                obtain(blob, fragment_keys_sorted, n_fragments, 128);
                obtain(blob, fragment_point_indices_unsorted, n_fragments, 128);
                obtain(blob, fragment_point_indices_sorted, n_fragments, 128);
                // using cub::DoubleBuffer halves the number of auxiliary memory required
                buffers.fragment_keys = cub::DoubleBuffer(fragment_keys_unsorted, fragment_keys_sorted);
                buffers.fragment_point_indices = cub::DoubleBuffer(fragment_point_indices_unsorted, fragment_point_indices_sorted);
                cub::DeviceRadixSort::SortPairs(
                    nullptr, buffers.cub_workspace_size,
                    buffers.fragment_keys,
                    buffers.fragment_point_indices,
                    n_fragments
                );
                obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
                return buffers;
            }
        };

        struct PerPixelBuffers {
            uint2* ranges;
    
            static PerPixelBuffers from_blob(char*& blob, size_t n_pixels) {
                PerPixelBuffers buffers;
                obtain(blob, buffers.ranges, n_pixels, 128);
                return buffers;
            }
        };
            
    }

}
