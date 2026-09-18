// Direct3D 11 memory.
//
// Separate from `memory::dx12`, and not for tidiness — but not for the reason it is tempting to
// give either. D3D11 DOES have a CUDA external-memory path, the same shape as D3D12's: a shared
// handle imported once, yielding a device pointer that stays valid. (There is also a legacy
// `cudaGraphicsD3D11RegisterResource` route with a map/unmap around every use; this module does not
// use it, because the external-memory path is better and matches D3D12.)
//
// What genuinely differs, and why these are two namespaces:
//   * different CUDA import types — `D3D11Resource` / `D3D11ResourceKmt` against
//     `D3D12Resource` / `D3D12Heap`;
//   * different handle kinds — D3D11 distinguishes an NT handle from the older global KMT handle,
//     D3D12 distinguishes a committed resource from a heap;
//   * different dedicated rule — EVERY D3D11 resource imports as dedicated, where D3D12 requires it
//     only for a committed resource.
//
// Folding them together would mean branching on those inside every function that touches either.
//
// This is the compatibility path; D3D12 is what a current engine uses.
#pragma once

#include <RadFiled3D/nn/backends/dx11.hpp>
#include <RadFiled3D/nn/memory/memory_ref.hpp>
#include <RadFiled3D/nn/memory/memory_ref.hpp>

namespace RadFiled3D::nn::memory::dx11 {

class MemoryRef final : public memory::MemoryRef {
public:
    /// `resource` is an `ID3D11Resource*` reinterpreted as an integer. The caller keeps the COM
    /// reference; this does not AddRef.
    MemoryRef(std::uint64_t resource, std::uint64_t size_bytes) noexcept
        : resource_(resource), size_bytes_(size_bytes) {}

    Domain get_domain() const noexcept override { return Domain::D3D11; }
    std::uint64_t get_size_bytes() const noexcept override { return size_bytes_; }

    /// Cast back to `ID3D11Resource*` on the caller's side.
    std::uint64_t get_address() const noexcept override { return resource_; }

    std::uint64_t get_resource() const noexcept { return resource_; }

private:
    std::uint64_t resource_;
    std::uint64_t size_bytes_;
};


}  // namespace RadFiled3D::nn::memory::dx11
