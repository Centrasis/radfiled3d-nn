#include <RadFiled3D/nn/backends/rocm.hpp>

#include <RadFiled3D/nn/backends/compute_backend.hpp>

namespace RadFiled3D::nn::rocm {

bool available() noexcept {
#ifdef RFNN_WITH_ROCM
    return true;
#else
    return false;
#endif
}

/// HIP's external-memory API mirrors CUDA's, so the same graphics allocations are reachable from an
/// AMD card — neither graphics backend privileges a vendor (R-G2).
class Backend final : public ComputeBackend {
public:
    nn::Backend get_backend() const noexcept override { return nn::Backend::Rocm; }
    std::string_view get_feature() const noexcept override { return "rocm"; }
    bool available() const noexcept override { return rocm::available(); }
    memory::Domain get_memory_domain() const noexcept override { return memory::Domain::Hip; }

    bool can_import_from(memory::Domain graphics) const noexcept override {
        switch (graphics) {
            case memory::Domain::Vulkan:
            case memory::Domain::D3D11:
            case memory::Domain::D3D12: return true;
            default: return false;
        }
    }

    std::shared_ptr<memory::MemoryRef> import_external_memory(const memory::ExternalOrigin&) const override {
        throw Exception::feature_disabled("rocm", "importing a graphics allocation");
    }

    /// Nothing of ROCm is compiled in here, so no device is reachable.
    int get_device_count() const noexcept override { return 0; }

    std::string_view get_block_target() const noexcept override { return "rocm"; }

    /// ROCm has no compiled-in code here at all (there is no ONNX Runtime release to fetch for it),
    /// so the architecture is unknown and a specialised `hsaco` block is never selected. An
    /// `ORT_ROOT` build on an AMD card is where this would report `gfx1100`.
    std::string get_device_arch(int) const noexcept override { return {}; }

    /// Nothing of ROCm is compiled in here (no ONNX Runtime release exists to fetch for it), so
    /// there is no HIP allocator to hand out. An `ORT_ROOT` build on an AMD card is where this gains
    /// a `hipMalloc` the way `memory/cuda.cpp` has a `cudaMalloc`.
    std::shared_ptr<memory::MemoryRef> allocate(std::uint64_t, int) const override {
        throw Exception::feature_disabled("rocm", "allocating HIP device memory");
    }

    void copy_within(memory::MemoryRef&, std::uint64_t, const memory::MemoryRef&, std::uint64_t,
                     std::uint64_t) const override {
        throw Exception::feature_disabled("rocm", "copying within HIP device memory");
    }

    void download(void*, const memory::MemoryRef&, std::uint64_t) const override {
        throw Exception::feature_disabled("rocm", "reading device memory back");
    }

    void upload(memory::MemoryRef&, const void*, std::uint64_t) const override {
        throw Exception::feature_disabled("rocm", "uploading to HIP device memory");
    }

    bool can_launch_kernels() const noexcept override { return false; }

    std::unique_ptr<Stage> make_kernel_stage(const deploy::Composition&, const deploy::Stage& stage,
                                             std::string_view kind,
                                             deploy::byte_view, const deploy::KernelLaunch&,
                                             int) const override {
        throw Exception::feature_disabled("rocm", "running stage `" + stage.name + "`, which is " +
                                                      std::string(kind));
    }

    std::string_view get_ort_allocator_name() const noexcept override { return "Hip"; }
};

const ComputeBackend& compute_backend() {
    static const Backend backend;
    return backend;
}

}  // namespace RadFiled3D::nn::rocm
