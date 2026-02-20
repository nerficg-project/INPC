#pragma once

#include "pcg32.h"
#include "helper_math.h"
#include <tuple>
#include <torch/extension.h>
#include <cstdint>

namespace inpc::sampling {

    torch::Tensor compute_viewpoint_weights_wrapper(
        const torch::Tensor& centers,
        const torch::Tensor& levels,
        const torch::Tensor& weights,
        const torch::Tensor& w2c,
        const uint width,
        const uint height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const float initial_size);

    struct ProbabilityFieldSampler {
        public:
            explicit ProbabilityFieldSampler(int64_t seed) : rng(seed) {};
            ~ProbabilityFieldSampler() = default;

            std::tuple<torch::Tensor, torch::Tensor> generate_training_samples_wrapper(
                const torch::Tensor& centers,
                const torch::Tensor& levels,
                const torch::Tensor& weights,
                const torch::Tensor& w2c,
                const uint n_samples,
                const uint width,
                const uint height,
                const float focal_x,
                const float focal_y,
                const float center_x,
                const float center_y,
                const float near_plane,
                const float far_plane,
                const float initial_size);

            torch::Tensor generate_samples_wrapper(
                const torch::Tensor& centers,
                const torch::Tensor& levels,
                const torch::Tensor& weights,
                const torch::Tensor& w2c,
                const uint n_samples,
                const uint width,
                const uint height,
                const float focal_x,
                const float focal_y,
                const float center_x,
                const float center_y,
                const float near_plane,
                const float far_plane,
                const float initial_size);

            torch::Tensor generate_expected_samples_wrapper(
                const torch::Tensor& centers,
                const torch::Tensor& levels,
                const torch::Tensor& weights,
                const torch::Tensor& w2c,
                const uint n_samples,
                const uint n_multi,
                const uint width,
                const uint height,
                const float focal_x,
                const float focal_y,
                const float center_x,
                const float center_y,
                const float near_plane,
                const float far_plane,
                const float initial_size);

        protected:
            pcg32 rng;
    };

}
