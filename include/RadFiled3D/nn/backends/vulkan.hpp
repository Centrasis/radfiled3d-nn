// `RadFiled3D::nn::vulkan` — compiled in with RFNN_WITH_VULKAN.
//
// The probe is declared and defined UNCONDITIONALLY, so a build without the option answers
// `false` instead of failing to link. That is what lets every API that needs this backend keep
// its full surface and throw `FeatureDisabled` naming the option.
#pragma once

namespace RadFiled3D::nn::vulkan {

/// Non-zero when this build has the backend compiled in.
bool available() noexcept;

}  // namespace RadFiled3D::nn::vulkan
