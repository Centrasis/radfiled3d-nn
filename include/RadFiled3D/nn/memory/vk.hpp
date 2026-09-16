// Vulkan memory.
//
// A Vulkan reference needs TWO handles, which is the concrete reason `MemoryRef` is specialised
// rather than flattened: the `VkBuffer` a shader binds, and the `VkDeviceMemory` it is bound to,
// which is what actually gets exported and imported. Both are 64-bit Vulkan handles, so no
// vulkan.h is needed here.
//
// Buffers bind directly. Images do not: a `VkImage`'s layout is implementation-defined (tiled and
// swizzled by the driver), so writing one needs a relayout copy through a linear staging buffer
// rather than a raw device pointer.
#pragma once

#include <RadFiled3D/nn/memory/external_memory.hpp>
#include <RadFiled3D/nn/memory/memory_ref.hpp>

namespace RadFiled3D::nn::memory::vk {

class MemoryRef final : public memory::MemoryRef {
public:
    /// `buffer` is a `VkBuffer`, `device_memory` the `VkDeviceMemory` it is bound to.
    MemoryRef(std::uint64_t buffer, std::uint64_t device_memory, std::uint64_t size_bytes,
              std::uint64_t offset_bytes = 0) noexcept
        : buffer_(buffer), device_memory_(device_memory), size_bytes_(size_bytes), offset_bytes_(offset_bytes) {}

    Domain get_domain() const noexcept override { return Domain::Vulkan; }
    std::uint64_t get_size_bytes() const noexcept override { return size_bytes_; }
    std::uint64_t get_offset_bytes() const noexcept override { return offset_bytes_; }

    std::uint64_t get_address() const noexcept override { return buffer_; }

    std::uint64_t get_buffer() const noexcept { return buffer_; }
    /// The allocation behind the buffer — this is what is exported to, and imported by, a compute
    /// backend, never the `VkBuffer` itself.
    std::uint64_t get_device_memory() const noexcept { return device_memory_; }

private:
    std::uint64_t buffer_;
    std::uint64_t device_memory_;
    std::uint64_t size_bytes_;
    std::uint64_t offset_bytes_;
};

/// Vulkan interop. The engine exports its `VkDeviceMemory` as an opaque handle
/// (`..._OPAQUE_FD_BIT` on Linux, `..._OPAQUE_WIN32_BIT` on Windows); the compute backend imports
/// it — `cudaImportExternalMemory` on NVIDIA, `hipImportExternalMemory` on AMD — and inference
/// writes the engine's memory in place. Neither vendor is privileged: the path is the same and only
/// the import call differs.
class ExternalMemory final : public memory::ExternalMemory {
public:
    Domain get_domain() const noexcept override { return Domain::Vulkan; }
    bool supports(Backend compute) const noexcept override;
    std::shared_ptr<memory::MemoryRef> import_buffer(const ExternalBuffer& buffer, Backend compute) override;
    ExternalBuffer export_buffer(std::uint64_t size_bytes, Backend compute) override;
};

}  // namespace RadFiled3D::nn::memory::vk
