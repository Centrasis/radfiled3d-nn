#include <RadFiled3D/nn/backends/directml.hpp>

#include <RadFiled3D/nn/backends/compute_backend.hpp>
#include <RadFiled3D/nn/backends/cpu.hpp>
#include <RadFiled3D/nn/memory/dx12.hpp>

namespace RadFiled3D::nn::directml {

bool available() noexcept {
#ifdef RFNN_WITH_DIRECTML
    return true;
#else
    return false;
#endif
}

/// DirectML runs on D3D12, so a D3D12 resource needs no import at all; a Vulkan one has no path to it.
class Backend final : public ComputeBackend {
public:
    nn::Backend get_backend() const noexcept override { return nn::Backend::DirectMl; }
    std::string_view get_feature() const noexcept override { return "directml"; }
    bool available() const noexcept override { return directml::available(); }
    memory::Domain get_memory_domain() const noexcept override { return memory::Domain::D3D12; }

    bool can_import_from(memory::Domain graphics) const noexcept override {
        switch (graphics) {
            // A D3D12 resource already IS this provider's memory — there is nothing to import, which is why
            // the pairing is possible but the import below is not an import.
            case memory::Domain::D3D12: return true;
            default: return false;
        }
    }

    /// THE NO-IMPORT CASE. DirectML runs on D3D12, so a D3D12 resource already *is* this backend's
    /// memory — there is nothing to import, and the reference stays in `Domain::D3D12` rather than
    /// being converted into some compute domain. That is the whole point of the pairing: on Windows
    /// it is the cheapest path there is, because it does no work at all.
    std::shared_ptr<memory::MemoryRef> import_external_memory(
        const memory::ExternalOrigin& buffer) const override {
        const auto* native = std::get_if<memory::D3D12NativeResource>(&buffer.handle);
        if (native == nullptr)
            // A SHARED handle would have to be opened with ID3D12Device::OpenSharedHandle before
            // DirectML could use it — a Direct3D call this library does not make, and does not need
            // to: the caller owns the device, so it owns the resource pointer already.
            throw Exception::invalid_argument(
                "DirectML takes the ID3D12Resource itself (memory::D3D12NativeResource), not a shared "
                "handle: it executes on D3D12, so nothing is shared and nothing is imported");
        if (native->resource == nullptr)
            throw Exception::invalid_argument("external memory: null ID3D12Resource");
        if (buffer.mapped_bytes() == 0)
            throw Exception::invalid_argument("external memory: the resource maps zero bytes");
        // Wrapping, not importing — and so nothing to release either: the reference owns no mapping
        // because none was made.
        return std::make_shared<memory::dx12::MemoryRef>(
            reinterpret_cast<std::uintptr_t>(native->resource), buffer.mapped_bytes(),
            buffer.offset_bytes);
    }

        /// ORT knows this provider's allocator as "DML".
    ///
    /// Naming it is only safe because the session does the wrapping step that goes with it: ORT's
    /// DirectML provider will not take a raw `ID3D12Resource*`, so the resource is turned into an
    /// opaque DML allocation with `OrtDmlApi::CreateGPUAllocationFromD3DResource` before it reaches
    /// a tensor (see `src/backends/onnx.cpp`). Without that, this name would let a session bind the
    /// raw pointer and compute silently on the wrong address — so the two must land together.
    /// DirectML runs on whatever D3D12 adapter the provider is given; enumerating adapters
    /// would need DXGI, which this build deliberately does not link.
    int get_device_count() const noexcept override { return 1; }

    /// Vendor-neutral, so there is no family a package could have compiled a block for.
    std::string_view get_block_target() const noexcept override { return {}; }

    /// DirectML is vendor-neutral by design: it runs on whatever D3D12 device is there, so there
    /// is no architecture name a package could have compiled a kernel for.
    std::string get_device_arch(int) const noexcept override { return {}; }

    /// A DirectML allocation is a D3D12 resource wrapped by the provider, obtained through the DML
    /// provider API rather than a plain allocator — which is a different shape from `cudaMalloc` and
    /// is not wired up. Composed models therefore run on this backend with HOST intermediates.
    std::shared_ptr<memory::MemoryRef> allocate(std::uint64_t bytes, int) const override {
        return memory::host::allocate(bytes);
    }

    void copy_within(memory::MemoryRef& dst, std::uint64_t dst_offset, const memory::MemoryRef& src,
                     std::uint64_t src_offset, std::uint64_t bytes) const override {
        cpu::compute_backend().copy_within(dst, dst_offset, src, src_offset, bytes);
    }

    void upload(memory::MemoryRef& dst, const void* source, std::uint64_t bytes) const override {
        cpu::compute_backend().upload(dst, source, bytes);
    }

    bool can_launch_kernels() const noexcept override { return false; }

    std::unique_ptr<Stage> make_kernel_stage(const deploy::Composition&, const deploy::Stage& stage,
                                             std::string_view kind,
                                             deploy::byte_view, const deploy::KernelLaunch&,
                                             int) const override {
        throw Exception::feature_disabled("directml", "running stage `" + stage.name + "`, which is " +
                                                          std::string(kind));
    }

    std::string_view get_ort_allocator_name() const noexcept override { return "DML"; }
};

const ComputeBackend& compute_backend() {
    static const Backend backend;
    return backend;
}

}  // namespace RadFiled3D::nn::directml
