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
    ///
    /// This form DENOTES the resource and nothing more. It cannot be adopted: no compute backend
    /// runs on D3D11 — DirectML is D3D12 — so unlike a `dx12::MemoryRef`, which DirectML consumes
    /// natively, a bare D3D11 reference has nobody to hand it to. Use it to describe an engine's
    /// allocation; use the second form to let a session write into it.
    MemoryRef(std::uint64_t resource, std::uint64_t size_bytes, std::uint64_t offset_bytes = 0) noexcept
        : resource_(resource), size_bytes_(size_bytes), offset_bytes_(offset_bytes) {}

    /// The SHAREABLE form: the resource, plus the handle a compute backend imports it through.
    ///
    /// This is the shape an engine that owns the D3D11 device hands over — Unity, say. The
    /// reference stays in `Domain::D3D11` and still denotes the `ID3D11Resource*`, so the engine
    /// can go on using it, while `ComputeBackend::adopt` has the origin it needs to import the same
    /// allocation as a CUDA pointer. One object describes both views of one piece of memory.
    MemoryRef(std::uint64_t resource, ExternalOrigin origin) noexcept
        : resource_(resource), size_bytes_(origin.mapped_bytes()),
          offset_bytes_(origin.offset_bytes), origin_(std::move(origin)) {}

    Domain get_domain() const noexcept override { return Domain::D3D11; }
    std::uint64_t get_size_bytes() const noexcept override { return size_bytes_; }
    std::uint64_t get_offset_bytes() const noexcept override { return offset_bytes_; }

    /// Cast back to `ID3D11Resource*` on the caller's side.
    std::uint64_t get_address() const noexcept override { return resource_; }

    std::uint64_t get_resource() const noexcept { return resource_; }

    const Origin& get_origin() const noexcept override { return origin_; }

    /// D3D11 shares NT and KMT handles, and the exporter keeps both — nothing is consumed by an
    /// import the way a POSIX descriptor is. The flag is still recorded so a second adoption of the
    /// same reference reads as the repeat it is.
    void mark_origin_consumed() noexcept override {
        if (auto* external = std::get_if<ExternalOrigin>(&origin_)) external->consumed = true;
    }

    /// Whether a compute backend can import this reference at all.
    bool is_shareable() const noexcept { return std::holds_alternative<ExternalOrigin>(origin_); }

private:
    std::uint64_t resource_;
    std::uint64_t size_bytes_;
    std::uint64_t offset_bytes_ = 0;
    Origin origin_{};
};


}  // namespace RadFiled3D::nn::memory::dx11
