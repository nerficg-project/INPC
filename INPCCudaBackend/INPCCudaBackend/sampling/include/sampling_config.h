#pragma once

#include "helper_math.h"

#define DEF inline constexpr

namespace inpc::sampling::config {
    DEF bool debug = false;
    DEF uint block_size_compute_weights = 256;
    DEF uint block_size_normalize_weights = 256;
    DEF uint block_size_create_samples = 256;
    DEF uint block_size_compute_sample_counts = 256;
    DEF uint block_size_repeat_interleave_indices = 256;
    DEF uint block_size_create_expected_samples = 256;
}

namespace config = inpc::sampling::config;

#undef DEF
