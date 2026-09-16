// `rfnn::cuda` — CUDA, compiled in with RFNN_WITH_CUDA.
//
// Device facts only: which CUDA device is which, so memory and sessions can be put on the same one.
// The memory import lives in `memory/cuda.hpp`, the inference session in `backends/onnx.hpp` (CUDA
// is an ONNX Runtime execution provider, not a separate runtime).
//
// It also runs a stage that is COMPILED CODE rather than a graph: `make_kernel_stage` loads a
// package's `cuda_ptx` or `cuda_cubin` block through the CUDA **driver** API and launches it. That
// is why this links `libcuda` beside `libcudart` — `cuModuleLoadData` and `cuLaunchKernel` have no
// runtime-API equivalent. Still no nvcc and no device code of our own: the kernels come out of the
// package, so this compiles with the same toolchain as the rest of the library.
//
// Everything here is declared UNCONDITIONALLY, so a build without the option answers `false` / `-1`
// instead of failing to link. Both are C APIs with no C++ ABI surface to mismatch against an
// engine's.
#pragma once

#include <RadFiled3D/nn/backends/compute_backend.hpp>

#include <array>
#include <cstdint>
#include <string>

namespace RadFiled3D::nn::cuda {

/// Non-zero when this build has CUDA compiled in. Says nothing about a device being present.
bool available() noexcept;

/// Devices the CUDA runtime can see. 0 without CUDA, and 0 on a machine with no usable driver —
/// so this is also the "is there a GPU" probe.
int get_device_count() noexcept;

/// The device's UUID, all-zero if there is no such device. The same 16 bytes Vulkan reports as
/// `VkPhysicalDeviceIDProperties::deviceUUID` for the same GPU, which is what makes the two APIs
/// agree on identity.
std::array<std::uint8_t, 16> get_device_uuid(int device) noexcept;

/// The CUDA ordinal for a Vulkan/D3D12 device UUID, or -1 if no device matches.
///
/// An all-zero UUID means "whichever device is current", which keeps the single-GPU case free of
/// ceremony; it returns the current device rather than guessing 0, so an application that already
/// chose a device is respected.
int get_device_for_uuid(const std::array<std::uint8_t, 16>& uuid) noexcept;

/// Human-readable name, for diagnostics. Empty if there is no such device.
std::string get_device_name(int device);

/// Make `device` current on this thread. Throws `FeatureDisabled` without CUDA, `InvalidArgument`
/// if the device does not exist.
void set_device(int device);

/// CUDA as a `ComputeBackend`. The only thing outside these files that should ever be needed.
const ComputeBackend& compute_backend();

}  // namespace RadFiled3D::nn::cuda
