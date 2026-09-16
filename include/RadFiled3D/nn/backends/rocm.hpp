// `RadFiled3D::nn::rocm` — compiled in with RFNN_WITH_ROCM.
//
// The probe is declared and defined UNCONDITIONALLY, so a build without the option answers
// `false` instead of failing to link. That is what lets every API that needs this backend keep
// its full surface and throw `FeatureDisabled` naming the option.
#pragma once

#include <RadFiled3D/nn/backends/compute_backend.hpp>

namespace RadFiled3D::nn::rocm {

/// Non-zero when this build has the backend compiled in.
bool available() noexcept;

/// Rocm as a `ComputeBackend`.
const ComputeBackend& compute_backend();

}  // namespace RadFiled3D::nn::rocm
