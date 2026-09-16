// `RadFiled3D::nn::onnx` — the ONNX Runtime, fetched on demand by CMake and compiled in with RFNN_WITH_ONNX.
//
// Declared and defined unconditionally, so a build without the option answers `false` / `nullopt`
// instead of failing to link.
#pragma once

#include <RadFiled3D/nn/core/session.hpp>

#include <memory>
#include <optional>
#include <string>

namespace RadFiled3D::nn::onnx {

/// Non-zero when this build has the ONNX Runtime linked in.
bool available() noexcept;

/// The linked runtime's version string, or `nullopt` without ONNX support. Reading it calls into
/// the library, so a value here proves the fetched runtime is genuinely linked and loadable.
std::optional<std::string> version();

/// Build a session for `package` on an ONNX Runtime execution provider.
///
/// Every backend this library has is an ORT execution provider — CPU, CUDA, TensorRT, ROCm,
/// DirectML — so there is ONE session implementation and the backends differ only in which provider
/// is appended and which memory domain its bindings accept. Four copies of the same binding logic is
/// what the alternative would cost.
///
/// `device` is the ordinal the session executes on, and everything uses it: the execution provider,
/// the intermediate buffers, a kernel stage's module and launches, and the weights uploaded for it.
/// -1 means "the default device". Memory bound from a DIFFERENT device is refused rather than
/// faulted on, so `nn::load(package, backend, Device::of(*imported))` is the way a renderer keeps
/// the session and its buffer on one card.
///
/// Throws `FeatureDisabled` when the provider was not compiled in.
std::unique_ptr<InferenceSession> load(deploy::Package package, Backend backend, int device = -1);

}  // namespace RadFiled3D::nn::onnx
