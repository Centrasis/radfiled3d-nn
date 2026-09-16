// HIP / ROCm device memory (AMD).
//
// The AMD counterpart of `memory::cuda`: the same import paths exist through HIP's external-memory
// API, so a Vulkan or D3D12 allocation shared from an AMD card lands here. Neither graphics backend
// privileges a vendor (R-G2).
#pragma once

#include <RadFiled3D/nn/memory/memory_ref.hpp>

namespace RadFiled3D::nn::memory::hip {

class MemoryRef final : public memory::MemoryRef {
public:
    /// `device_ptr` is a `hipDeviceptr_t`.
    MemoryRef(std::uint64_t device_ptr, std::uint64_t size_bytes) noexcept
        : device_ptr_(device_ptr), size_bytes_(size_bytes) {}

    Domain get_domain() const noexcept override { return Domain::Hip; }
    std::uint64_t get_size_bytes() const noexcept override { return size_bytes_; }

    std::uint64_t get_address() const noexcept override { return device_ptr_; }

    std::uint64_t get_device_ptr() const noexcept { return device_ptr_; }

private:
    std::uint64_t device_ptr_;
    std::uint64_t size_bytes_;
};

}  // namespace RadFiled3D::nn::memory::hip
