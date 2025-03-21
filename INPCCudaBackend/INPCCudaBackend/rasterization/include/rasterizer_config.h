#pragma once

#include "helper_math.h"

#define DEF inline constexpr

namespace inpc::rasterization::config {
    DEF float transmittance_threshold = 0.0001f;
    namespace training::forward {
        DEF bool debug = false;
        DEF uint block_size_preprocess = 256;
        DEF uint block_size_blend = 256;
    }
    namespace training::backward {
        DEF bool debug = false;
        DEF uint block_size_blend = 128;
        DEF uint block_size_convert_features = 256;
    }
    namespace inference::render {
        DEF uint features_per_point = 37; // 1 opacity + 9 SH coefficients * 4 channels
        DEF bool debug = false;
        DEF uint block_size_preprocess = 256;
        DEF uint block_size_blend = 256;
    }
    namespace inference::render_preextracted {
        DEF bool debug = false;
        DEF uint block_size_preprocess = 256;
        DEF uint block_size_create_fragments = 256;
        DEF uint block_size_find_ranges = 256;
        DEF uint block_size_blend = 256;
    }
}

namespace config = inpc::rasterization::config;

#undef DEF
