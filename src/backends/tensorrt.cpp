#include <RadFiled3D/nn/backends/tensorrt.hpp>

#include <RadFiled3D/nn/backends/compute_backend.hpp>
#include <RadFiled3D/nn/backends/cuda.hpp>

namespace RadFiled3D::nn::tensorrt {

bool available() noexcept {
#ifdef RFNN_WITH_TENSORRT
    return true;
#else
    return false;
#endif
}

/// TensorRT runs on CUDA and shares its memory entirely — it is a different ONNX Runtime execution
/// provider over the same allocations, not a different allocator. So everything memory-shaped is
/// delegated to the CUDA backend rather than restated here, and there is exactly one implementation
/// of the import.
class Backend final : public ComputeBackend {
public:
    nn::Backend get_backend() const noexcept override { return nn::Backend::TensorRt; }
    std::string_view get_feature() const noexcept override { return "tensorrt"; }
    bool available() const noexcept override { return tensorrt::available(); }

    memory::Domain get_memory_domain() const noexcept override {
        return cuda::compute_backend().get_memory_domain();
    }
    bool can_import_from(memory::Domain graphics) const noexcept override {
        return cuda::compute_backend().can_import_from(graphics);
    }
    std::shared_ptr<memory::MemoryRef> import_external_memory(
        const memory::ExternalOrigin& buffer) const override {
        return cuda::compute_backend().import_external_memory(buffer);
    }
    int get_device_count() const noexcept override {
        return cuda::compute_backend().get_device_count();
    }

    /// TensorRT is a provider on a CUDA device, not a hardware family of its own, so a
    /// `cuda_ptx` block runs under it exactly as it does under the CUDA provider.
    std::string_view get_block_target() const noexcept override {
        return cuda::compute_backend().get_block_target();
    }

    /// TensorRT runs on the CUDA device, so it reports the CUDA architecture.
    std::string get_device_arch(int device) const noexcept override {
        return cuda::compute_backend().get_device_arch(device);
    }

    /// TensorRT runs on the CUDA device, so its memory and its kernels are CUDA's.
    std::shared_ptr<memory::MemoryRef> allocate(std::uint64_t bytes, int device) const override {
        return cuda::compute_backend().allocate(bytes, device);
    }

    void copy_within(memory::MemoryRef& dst, std::uint64_t dst_offset, const memory::MemoryRef& src,
                     std::uint64_t src_offset, std::uint64_t bytes) const override {
        cuda::compute_backend().copy_within(dst, dst_offset, src, src_offset, bytes);
    }

    void upload(memory::MemoryRef& dst, const void* source, std::uint64_t bytes) const override {
        cuda::compute_backend().upload(dst, source, bytes);
    }

    void download(void* destination, const memory::MemoryRef& source,
                  std::uint64_t bytes) const override {
        cuda::compute_backend().download(destination, source, bytes);
    }

    bool can_launch_kernels() const noexcept override {
        return cuda::compute_backend().can_launch_kernels();
    }

    std::unique_ptr<Stage> make_kernel_stage(const deploy::Composition& composition, const deploy::Stage& stage,
                                             std::string_view kind,
                                             deploy::byte_view code, const deploy::KernelLaunch& launch,
                                             int device) const override {
        return cuda::compute_backend().make_kernel_stage(composition, stage, kind, code, launch,
                                                         device);
    }

    std::string_view get_ort_allocator_name() const noexcept override {
        return cuda::compute_backend().get_ort_allocator_name();
    }
};

const ComputeBackend& compute_backend() {
    static const Backend backend;
    return backend;
}

}  // namespace RadFiled3D::nn::tensorrt
