#include <RadFiled3D/nn/core/session.hpp>

#include <RadFiled3D/nn/backends/compute_backend.hpp>
#include <RadFiled3D/nn/backends/cpu.hpp>
#include <RadFiled3D/nn/backends/cuda.hpp>
#include <RadFiled3D/nn/backends/directml.hpp>
#include <RadFiled3D/nn/backends/onnx.hpp>
#include <RadFiled3D/nn/backends/rocm.hpp>
#include <RadFiled3D/nn/backends/tensorrt.hpp>

namespace RadFiled3D::nn {

std::string_view to_string(Backend backend) noexcept {
    switch (backend) {
        case Backend::Cpu: return "cpu";
        case Backend::Cuda: return "cuda";
        case Backend::TensorRt: return "tensorrt";
        case Backend::Rocm: return "rocm";
        case Backend::DirectMl: return "directml";
    }
    return "?";
}

Backend backend_from_name(std::string_view name) {
    for (const Backend b : kBackends)
        if (to_string(b) == name) return b;
    throw Exception::invalid_argument("unknown backend `" + std::string(name) +
                                  "`; expected one of cpu, cuda, tensorrt, rocm, directml");
}

std::string_view get_required_feature(Backend backend) noexcept {
    return get_compute_backend(backend).get_feature();
}

const ComputeBackend& get_compute_backend(Backend backend) {
    // The ONE place that enumerates the backends. It sits beside `to_string` and `backend_from_name`
    // because CLAUDE.md already puts the backend vocabulary here and this is part of that
    // vocabulary — everywhere else asks a `ComputeBackend` what it can do instead of asking which
    // one it is.
    switch (backend) {
        case Backend::Cpu: return cpu::compute_backend();
        case Backend::Cuda: return cuda::compute_backend();
        case Backend::TensorRt: return tensorrt::compute_backend();
        case Backend::Rocm: return rocm::compute_backend();
        case Backend::DirectMl: return directml::compute_backend();
    }
    throw Exception::invalid_argument("unknown backend");
}

bool is_available(Backend backend) noexcept { return get_compute_backend(backend).available(); }

bool can_share_graphics_memory(Backend backend) noexcept { return backend != Backend::Cpu; }

std::unique_ptr<InferenceSession> load(deploy::Package package, Backend backend, Device device) {
    package.validate();
    // Name what is ACTUALLY missing. Every backend is an ONNX Runtime execution provider, so with
    // ORT absent the answer is "onnx" whichever backend was asked for — telling a caller to enable
    // `cpu`, which is on by default, would send them to fix the wrong thing.
    if (!onnx::available())
        throw Exception::feature_disabled(
            "onnx", "building an inference session (every backend is an ONNX Runtime execution provider)");
    if (!is_available(backend))
        throw Exception::feature_disabled(get_required_feature(backend), "building an inference session");
    // Every backend is an ONNX Runtime execution provider, so the dispatch is one call; which
    // provider gets appended is the backend's business, inside onnx::load.
    return onnx::load(std::move(package), backend, device.get_index());
}

}  // namespace RadFiled3D::nn
