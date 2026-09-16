#include <RadFiled3D/nn/backends/cpu.hpp>

#include <RadFiled3D/nn/backends/onnx.hpp>
#include <RadFiled3D/nn/exception.hpp>

#include <cstring>

namespace RadFiled3D::nn::cpu {

bool available() noexcept {
#ifdef RFNN_WITH_CPU
    // The CPU provider is ONNX Runtime's own, so it cannot be present without ORT however the
    // option is set — the option decides whether this build OFFERS it, not whether ORT has it.
    return onnx::available();
#else
    return false;
#endif
}

namespace {

class Backend final : public ComputeBackend {
public:
    nn::Backend get_backend() const noexcept override { return nn::Backend::Cpu; }
    std::string_view get_feature() const noexcept override { return "cpu"; }
    bool available() const noexcept override { return cpu::available(); }
    memory::Domain get_memory_domain() const noexcept override { return memory::Domain::Host; }

    /// Nothing. Sharing graphics memory with the CPU provider is not a build question — there is no
    /// mechanism, on any platform, so callers get `UnsupportedInterop` rather than a hint to enable
    /// something that would not help.
    bool can_import_from(memory::Domain) const noexcept override { return false; }

    std::shared_ptr<memory::MemoryRef> import_external_memory(const memory::ExternalBuffer&) const override {
        throw Exception::unsupported_interop("graphics", "cpu");
    }

    /// Empty: host memory is ORT's default allocator, named by `CreateCpu` rather than by string.
    /// One "device": the process. There is nothing to choose between.
    int get_device_count() const noexcept override { return 1; }

    /// No compiled block targets the CPU: a kernel is device code by definition.
    std::string_view get_block_target() const noexcept override { return {}; }

    /// The CPU provider has no architecture to match a specialised block against.
    std::string get_device_arch(int) const noexcept override { return {}; }

    /// Ordinary process memory: the CPU provider's own domain, so an intermediate needs no
    /// transfer at all.
    std::shared_ptr<memory::MemoryRef> allocate(std::uint64_t bytes, int) const override {
        return memory::host::allocate(bytes);
    }

    void copy_within(memory::MemoryRef& dst, std::uint64_t dst_offset, const memory::MemoryRef& src,
                     std::uint64_t src_offset, std::uint64_t bytes) const override {
        if (bytes == 0) return;
        dst.require_domain(memory::Domain::Host);
        src.require_domain(memory::Domain::Host);
        if (dst_offset + bytes > dst.get_size_bytes() || src_offset + bytes > src.get_size_bytes())
            throw Exception::invalid_argument("host copy would run past the end of an allocation");
        std::memcpy(reinterpret_cast<std::byte*>(static_cast<std::uintptr_t>(dst.get_address())) + dst_offset,
                    reinterpret_cast<const std::byte*>(static_cast<std::uintptr_t>(src.get_address())) +
                        src_offset,
                    static_cast<std::size_t>(bytes));
    }

    void upload(memory::MemoryRef& dst, const void* source, std::uint64_t bytes) const override {
        if (bytes == 0) return;
        dst.require_domain(memory::Domain::Host);
        if (bytes > dst.get_size_bytes())
            throw Exception::invalid_argument("upload would run past the end of an allocation");
        std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dst.get_address())), source,
                    static_cast<std::size_t>(bytes));
    }

    /// A compiled kernel is device code; there is nothing here to run it on.
    bool can_launch_kernels() const noexcept override { return false; }

    std::unique_ptr<Stage> make_kernel_stage(const deploy::Composition&, const deploy::Stage& stage,
                                             std::string_view kind,
                                             deploy::byte_view, const deploy::KernelLaunch&,
                                             int) const override {
        throw Exception::feature_disabled(
            "cuda", "running stage `" + stage.name + "`, which is " + std::string(kind) +
                        " — the CPU provider executes graphs, not device kernels");
    }

    std::string_view get_ort_allocator_name() const noexcept override { return ""; }
};

}  // namespace

const ComputeBackend& compute_backend() {
    static const Backend backend;
    return backend;
}

}  // namespace RadFiled3D::nn::cpu
