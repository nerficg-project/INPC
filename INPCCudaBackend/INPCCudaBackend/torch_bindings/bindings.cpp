#include <torch/extension.h>
#include "rasterization_api.h"
#include "sampling_api.h"

namespace rasterization_api = inpc::rasterization;
namespace sampling_api = inpc::sampling;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // rasterization
    m.def("forward", &rasterization_api::forward_wrapper);
    m.def("backward", &rasterization_api::backward_wrapper);
    m.def("render", &rasterization_api::render_default_wrapper);
    m.def("render_preextracted", &rasterization_api::render_preextracted_wrapper);
    // sampling
    m.def("compute_viewpoint_weights_cuda", &sampling_api::compute_viewpoint_weights_wrapper);
    pybind11::class_<sampling_api::ProbabilityFieldSampler>(m, "ProbabilityFieldSamplerCUDA")
        .def(py::init<int64_t>())
        .def("generate_samples", &sampling_api::ProbabilityFieldSampler::generate_samples_wrapper)
        .def("generate_expected_samples", &sampling_api::ProbabilityFieldSampler::generate_expected_samples_wrapper);
}
