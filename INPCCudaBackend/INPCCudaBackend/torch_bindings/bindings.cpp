#include <torch/extension.h>
#include "rasterization_api.h"
#include "sampling_api.h"
#include "misc_api.h"

namespace rasterization_api = inpc::rasterization;
namespace sampling_api = inpc::sampling;
namespace misc_api = inpc::misc;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // rasterization
    m.def("forward", &rasterization_api::forward_wrapper);
    m.def("backward", &rasterization_api::backward_wrapper);
    m.def("render", &rasterization_api::inference_wrapper);
    m.def("render_preextracted", &rasterization_api::inference_preextracted_wrapper);
    // sampling
    m.def("compute_viewpoint_weights_cuda", &sampling_api::compute_viewpoint_weights_wrapper);
    pybind11::class_<sampling_api::ProbabilityFieldSampler>(m, "ProbabilityFieldSamplerCUDA")
        .def(py::init<int64_t>())
        .def("generate_training_samples", &sampling_api::ProbabilityFieldSampler::generate_training_samples_wrapper)
        .def("generate_samples", &sampling_api::ProbabilityFieldSampler::generate_samples_wrapper)
        .def("generate_expected_samples", &sampling_api::ProbabilityFieldSampler::generate_expected_samples_wrapper);
    // misc
    m.def("compute_normalized_weight_decay_grads_cuda", &misc_api::compute_normalized_weight_decay_grads_wrapper);
    m.def("add_normalized_weight_decay_grads_cuda", &misc_api::add_normalized_weight_decay_grads_wrapper);
    m.def("spherical_contraction_cuda", &misc_api::spherical_contraction_wrapper);
    m.def("cauchy_loss_cuda", &misc_api::cauchy_loss_wrapper);
    m.def("cauchy_loss_backward_cuda", &misc_api::cauchy_loss_backward_wrapper);
}
