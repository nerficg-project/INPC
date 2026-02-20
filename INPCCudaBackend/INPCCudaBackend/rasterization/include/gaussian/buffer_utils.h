#pragma once

#include "helper_math.h"
#include <cub/cub.cuh>
#include <cstdint>

namespace inpc::rasterization::gaussian {

    // TODO: duplicate code (see utils.h)
    template <typename T>
    static void obtain(char*& blob, T*& ptr, std::size_t count, std::size_t alignment) {
        std::size_t offset = reinterpret_cast<std::uintptr_t>(blob) + alignment - 1 & ~(alignment - 1);
        ptr = reinterpret_cast<T*>(offset);
        blob = reinterpret_cast<char*>(ptr + count);
    }

    template<typename T, typename... Args> 
	size_t required(size_t P, Args... args){
		char* size = nullptr;
		T::from_blob(size, P, args...);
		return ((size_t)size) + 128;
	}

    struct PerPrimitiveBuffers {
        size_t cub_workspace_size;
        char* cub_workspace;
        cub::DoubleBuffer<uint> depth_keys;
        cub::DoubleBuffer<uint> primitive_indices;
        uint* n_touched_tiles;
        uint* offset;
        ushort4* screen_bounds;
        float2* screen_coords;
        float4* features_view;
        float* depth;
        float3* conic;
        float* opacity = nullptr;
        uint* n_visible_primitives;
        uint* n_instances;

        static PerPrimitiveBuffers from_blob(char*& blob, int n_primitives, bool store_opacity = false) {
            PerPrimitiveBuffers buffers;
            uint* depth_keys_current;
            obtain(blob, depth_keys_current, n_primitives, 128);
            uint* depth_keys_alternate;
            obtain(blob, depth_keys_alternate, n_primitives, 128);
            buffers.depth_keys = cub::DoubleBuffer<uint>(depth_keys_current, depth_keys_alternate);
            uint* primitive_indices_current;
            obtain(blob, primitive_indices_current, n_primitives, 128);
            uint* primitive_indices_alternate;
            obtain(blob, primitive_indices_alternate, n_primitives, 128);
            buffers.primitive_indices = cub::DoubleBuffer<uint>(primitive_indices_current, primitive_indices_alternate);
            obtain(blob, buffers.n_touched_tiles, n_primitives, 128);
            obtain(blob, buffers.offset, n_primitives, 128);
            obtain(blob, buffers.screen_bounds, n_primitives, 128);
            obtain(blob, buffers.screen_coords, n_primitives, 128);
            obtain(blob, buffers.features_view, n_primitives, 128);
            obtain(blob, buffers.depth, n_primitives, 128);
            obtain(blob, buffers.conic, n_primitives, 128);
            if (store_opacity) obtain(blob, buffers.opacity, n_primitives, 128);
            cub::DeviceScan::ExclusiveSum(
                nullptr, buffers.cub_workspace_size,
                buffers.offset, buffers.offset,
                n_primitives
            );
            size_t sorting_workspace_size;
            cub::DeviceRadixSort::SortPairs(
                nullptr, sorting_workspace_size,
                buffers.depth_keys, buffers.primitive_indices,
                n_primitives,
                0, 32
            );
            buffers.cub_workspace_size = max(buffers.cub_workspace_size, sorting_workspace_size);
            obtain(blob, buffers.n_visible_primitives, 1, 128);
            obtain(blob, buffers.n_instances, 1, 128);
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
            return buffers;
        }
    };

    struct PerInstanceBuffers {
        size_t cub_workspace_size;
        char* cub_workspace;
        cub::DoubleBuffer<ushort> keys;
        cub::DoubleBuffer<uint> primitive_indices;
    
        static PerInstanceBuffers from_blob(char*& blob, int n_instances, int end_bit) {
            PerInstanceBuffers buffers;
            ushort* keys_current;
            obtain(blob, keys_current, n_instances, 128);
            ushort* keys_alternate;
            obtain(blob, keys_alternate, n_instances, 128);
            buffers.keys = cub::DoubleBuffer<ushort>(keys_current, keys_alternate);
            uint* primitive_indices_current;
            obtain(blob, primitive_indices_current, n_instances, 128);
            uint* primitive_indices_alternate;
            obtain(blob, primitive_indices_alternate, n_instances, 128);
            buffers.primitive_indices = cub::DoubleBuffer<uint>(primitive_indices_current, primitive_indices_alternate);
            cub::DeviceRadixSort::SortPairs(
                nullptr, buffers.cub_workspace_size,
                buffers.keys, buffers.primitive_indices,
                n_instances,
                0, end_bit
            );
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
            return buffers;
        }
    };

    struct PerTileBuffers {
        uint2* instance_ranges;
    
        static PerTileBuffers from_blob(char*& blob, int n_tiles) {
            PerTileBuffers buffers;
            obtain(blob, buffers.instance_ranges, n_tiles, 128);
            return buffers;
        }
    };

}
