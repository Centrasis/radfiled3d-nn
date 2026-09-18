// Direct3D 12 memory.
//
// The primary Windows path, and the one an Unreal Engine 5 plugin uses. A D3D12 resource or heap is
// shared through the MODERN external-memory route: the engine creates it shared, hands over an NT
// handle, and the compute backend imports it once (`cudaImportExternalMemory` with
// `D3D12Resource`/`D3D12Heap`, or HIP's equivalent) to get a device pointer that stays valid.
//
// Under the DirectML execution provider there is no import at all: a D3D12 resource already IS the
// provider's native memory, so the reference is bound as-is.
//
// Kept apart from `memory::dx11` deliberately — see that header.
#pragma once

#include <RadFiled3D/nn/backends/dx12.hpp>
#include <RadFiled3D/nn/memory/memory_ref.hpp>
#include <RadFiled3D/nn/memory/memory_ref.hpp>

namespace RadFiled3D::nn::memory::dx12 {

class MemoryRef final : public memory::MemoryRef {
public:
    /// `resource` is an `ID3D12Resource*` reinterpreted as an integer. The caller keeps the COM
    /// reference; this does not AddRef.
    MemoryRef(std::uint64_t resource, std::uint64_t size_bytes, std::uint64_t offset_bytes = 0) noexcept
        : resource_(resource), size_bytes_(size_bytes), offset_bytes_(offset_bytes) {}

    Domain get_domain() const noexcept override { return Domain::D3D12; }
    std::uint64_t get_size_bytes() const noexcept override { return size_bytes_; }
    std::uint64_t get_offset_bytes() const noexcept override { return offset_bytes_; }

    /// Cast back to `ID3D12Resource*` on the caller's side.
    std::uint64_t get_address() const noexcept override { return resource_; }

    std::uint64_t get_resource() const noexcept { return resource_; }

private:
    std::uint64_t resource_;
    std::uint64_t size_bytes_;
    std::uint64_t offset_bytes_;
};


}  // namespace RadFiled3D::nn::memory::dx12
