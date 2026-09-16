// `rfnn::cpu` — the portable CPU execution provider.
//
// It has a file pair like every other backend even though it wraps no SDK, because "which backend
// am I" should be answerable the same way for all of them. What it wraps is the absence of a
// device: its memory is ordinary host memory and no graphics allocation can reach it.
//
// Availability follows ONNX Runtime — the CPU provider is ORT's, so a build without ORT has no CPU
// backend either, and saying so is more useful than claiming a backend that cannot run anything.
#pragma once

#include <RadFiled3D/nn/backends/compute_backend.hpp>

namespace RadFiled3D::nn::cpu {

bool available() noexcept;

/// The CPU provider as a `ComputeBackend`.
const ComputeBackend& compute_backend();

}  // namespace RadFiled3D::nn::cpu
