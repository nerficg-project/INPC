#pragma once

#include "helper_math.h"

#define DEF inline constexpr

namespace inpc::rasterization::bilinear::config {
    // debugging constants
    DEF bool debug_forward = false;
    DEF bool debug_backward = false;
    DEF bool debug_inference = false;
    DEF bool debug_inference_preextracted = false;
    DEF bool debug_fast_inference = false;
    DEF bool debug_fast_inference_preextracted = false;
    // rendering constants
    DEF float transmittance_threshold = 1e-4f;
    DEF float one_minus_alpha_eps = 1e-8f;
    DEF float min_alpha_threshold = 1.0f / 255.0f; // 0.00392156862
    // block size constants
    DEF int block_size_preprocess = 256;
    DEF int block_size_create_instances = 256;
    DEF int block_size_extract_instance_ranges = 256;
    DEF int tile_width = 8;
    DEF int tile_height = 8;
    DEF int block_size_blend = tile_width * tile_height;
}

namespace config = inpc::rasterization::bilinear::config;

#undef DEF
