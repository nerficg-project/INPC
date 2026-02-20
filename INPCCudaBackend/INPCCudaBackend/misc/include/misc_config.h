#pragma once

#define DEF inline constexpr

namespace inpc::misc::config {
    DEF bool debug = false;
    DEF uint block_size_compute_normalized_weight_decay_grads = 256;
    DEF uint block_size_add_normalized_weight_decay_grads = 256;
    DEF uint block_size_spherical_contraction = 256;
    DEF uint block_size_cauchy_loss = 256;
    DEF uint block_size_cauchy_loss_backward = 256;
}

namespace config = inpc::misc::config;

#undef DEF
