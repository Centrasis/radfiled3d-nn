#include <RadFiled3D/nn/backends/cuda.hpp>

#include <RadFiled3D/nn/backends/compute_backend.hpp>
#include <RadFiled3D/nn/exception.hpp>
#include <RadFiled3D/nn/memory/cuda.hpp>

#ifdef RFNN_WITH_CUDA
// The DRIVER API as well as the runtime: `cuModuleLoadData` and `cuLaunchKernel` have no
// runtime-API equivalent, and running a package's own compiled code is what they are here for.
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <string>
#include <vector>

namespace RadFiled3D::nn::cuda {

bool available() noexcept {
#ifdef RFNN_WITH_CUDA
    return true;
#else
    return false;
#endif
}

int get_device_count() noexcept {
#ifdef RFNN_WITH_CUDA
    int count = 0;
    // A machine with the libraries but no usable driver returns an error here rather than 0, and
    // "no devices" is the honest answer in both cases.
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return count;
#else
    return 0;
#endif
}

std::array<std::uint8_t, 16> get_device_uuid([[maybe_unused]] int device) noexcept {
    std::array<std::uint8_t, 16> uuid{};
#ifdef RFNN_WITH_CUDA
    if (device < 0 || device >= get_device_count()) return uuid;
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, device) != cudaSuccess) return uuid;
    std::copy_n(reinterpret_cast<const std::uint8_t*>(props.uuid.bytes), uuid.size(), uuid.begin());
#endif
    return uuid;
}

int get_device_for_uuid([[maybe_unused]] const std::array<std::uint8_t, 16>& uuid) noexcept {
#ifdef RFNN_WITH_CUDA
    const bool unspecified = std::all_of(uuid.begin(), uuid.end(), [](std::uint8_t b) { return b == 0; });
    if (unspecified) {
        int current = 0;
        if (cudaGetDevice(&current) != cudaSuccess) return -1;
        return current;
    }
    for (int device = 0; device < get_device_count(); ++device)
        if (get_device_uuid(device) == uuid) return device;
#endif
    return -1;
}

std::string get_device_name([[maybe_unused]] int device) {
#ifdef RFNN_WITH_CUDA
    if (device < 0 || device >= get_device_count()) return {};
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, device) != cudaSuccess) return {};
    return props.name;
#else
    return {};
#endif
}

void set_device([[maybe_unused]] int device) {
#ifdef RFNN_WITH_CUDA
    if (device < 0 || device >= get_device_count())
        throw Exception::invalid_argument("CUDA device " + std::to_string(device) + " does not exist (" +
                                          std::to_string(get_device_count()) + " visible)");
    if (cudaSetDevice(device) != cudaSuccess)
        throw Exception::invalid_argument("cudaSetDevice(" + std::to_string(device) + ") failed");
#else
    throw Exception::feature_disabled("cuda", "selecting a CUDA device");
#endif
}

/// The `ComputeBackend` face of CUDA. Everything the rest of the library needs to know about CUDA
/// is answered from here, so no other file includes a CUDA header or names a CUDA type.
class Backend final : public ComputeBackend {
public:
    nn::Backend get_backend() const noexcept override { return nn::Backend::Cuda; }
    std::string_view get_feature() const noexcept override { return "cuda"; }
    bool available() const noexcept override { return cuda::available(); }
    memory::Domain get_memory_domain() const noexcept override { return memory::Domain::Cuda; }

    bool can_import_from(memory::Domain graphics) const noexcept override {
        // CUDA's external-memory API accepts a Vulkan allocation and a D3D11/D3D12 resource. It has
        // no path from host memory, and none from another compute backend's.
        switch (graphics) {
            case memory::Domain::Vulkan:
            case memory::Domain::D3D11:
            case memory::Domain::D3D12: return true;
            default: return false;
        }
    }

    std::shared_ptr<memory::MemoryRef> import_external_memory(
        const memory::ExternalOrigin& buffer) const override {
        // Resolving the device here is why a graphics backend never has to know what a CUDA ordinal
        // is: it hands over the UUID its own API gave it and gets back bindable memory.
        return memory::cuda::import_external_memory(buffer, get_device_for_uuid(buffer.device_uuid));
    }

        int get_device_count() const noexcept override { return cuda::get_device_count(); }

    std::string_view get_block_target() const noexcept override { return "cuda"; }

    /// PTX and cubin blocks are launched through the CUDA driver API; there is nothing here that
    /// only a graph runtime could do.
    bool can_launch_kernels() const noexcept override {
#ifdef RFNN_WITH_CUDA
        return true;
#else
        return false;
#endif
    }

    std::shared_ptr<memory::MemoryRef> allocate(std::uint64_t bytes, int device) const override {
        return memory::cuda::allocate(bytes, device);
    }

    void copy_within(memory::MemoryRef& dst, std::uint64_t dst_offset, const memory::MemoryRef& src,
                     std::uint64_t src_offset, std::uint64_t bytes) const override {
        memory::cuda::copy_device_to_device(dst, dst_offset, src, src_offset, bytes);
    }

    void upload([[maybe_unused]] memory::MemoryRef& dst, [[maybe_unused]] const void* source,
                [[maybe_unused]] std::uint64_t bytes) const override {
#ifndef RFNN_WITH_CUDA
        throw Exception::feature_disabled("cuda", "uploading to CUDA device memory");
#else
        if (bytes == 0) return;
        dst.require_domain(memory::Domain::Cuda);
        if (bytes > dst.get_size_bytes())
            throw Exception::invalid_argument("upload would run past the end of an allocation");
        // In the destination's own context, for the same reason the device-to-device copy is.
        if (dst.get_device_index() >= 0) cudaSetDevice(dst.get_device_index());
        if (cudaMemcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dst.get_address())), source,
                       static_cast<std::size_t>(bytes), cudaMemcpyHostToDevice) != cudaSuccess)
            throw Exception::invalid_argument("cudaMemcpy host-to-device failed");
#endif
    }

    std::unique_ptr<Stage> make_kernel_stage(const deploy::Composition& composition,
                                             const deploy::Stage& stage, std::string_view kind,
                                             deploy::byte_view code, const deploy::KernelLaunch& launch,
                                             int device) const override;

    /// `sm_<major><minor>`, the spelling `nvcc -arch` and a package's `target_arch` both use, so a
    /// block compiled for this device matches it by string equality and nothing has to parse it.
    std::string get_device_arch(int device) const noexcept override {
#ifndef RFNN_WITH_CUDA
        (void)device;
        return {};
#else
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, device < 0 ? 0 : device) != cudaSuccess) return {};
        return "sm_" + std::to_string(props.major) + std::to_string(props.minor);
#endif
    }

    std::string_view get_ort_allocator_name() const noexcept override { return "Cuda"; }
};

#ifdef RFNN_WITH_CUDA
namespace {

/// Report a driver-API failure with the call that produced it and what the driver said.
void check(CUresult result, const char* call, const std::string& stage) {
    if (result == CUDA_SUCCESS) return;
    const char* name = nullptr;
    const char* text = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &text);
    throw Exception::invalid_package("stage `" + stage + "`: " + call + " failed — " +
                                     (name != nullptr ? name : "?") + ": " +
                                     (text != nullptr ? text : "no description"));
}

/// Make the device's PRIMARY context current on this thread.
///
/// Deliberately not `cuCtxCreate`: ONNX Runtime's CUDA provider allocates in the primary context, so
/// a context of our own would put its tensors and our kernel launches in different address spaces —
/// and every pointer handed across would be valid in one and meaningless in the other, which the
/// driver reports as an unspecified launch failure rather than as the mistake it is.
void use_primary_context(int device) {
    if (cudaSetDevice(device) != cudaSuccess)
        throw Exception::invalid_argument("no CUDA device " + std::to_string(device));
    // Forces the runtime to create and bind the primary context if it has not already.
    if (cudaFree(nullptr) != cudaSuccess)
        throw Exception::invalid_argument("could not establish a CUDA context on device " +
                                          std::to_string(device));
}

/// A stage that is COMPILED CODE rather than a graph.
///
/// The module is loaded once, when the session is built, and the argument vector is assembled once,
/// when the grid is chosen. `run()` is then a single `cuLaunchKernel` against pointers that have not
/// moved — no allocation, no staging, no host round trip, which is the whole reason the
/// intermediates live in device memory.
class KernelStage final : public nn::Stage {
public:
    KernelStage(const deploy::Composition& composition, const deploy::Stage& declared,
                std::string_view kind, deploy::byte_view code, deploy::KernelLaunch launch, int device)
        : name_(declared.name), invocation_(declared.invocation), launch_(std::move(launch)),
          device_(device < 0 ? 0 : device) {
        // A kernel declares no tensors, so its arguments ARE its ports — reads then writes, in the
        // order the composition lists them. That is the contract in `docs/custom-code.md` §4, and it
        // is the only thing a blob of machine code can be held to.
        for (const auto& port : declared.reads) {
            inputs_.push_back(port.tensor);
            input_elements_.push_back(elements_of(composition, port.buffer, declared.name));
        }
        for (const auto& port : declared.writes) {
            outputs_.push_back(port.tensor);
            output_elements_.push_back(elements_of(composition, port.buffer, declared.name));
        }

        use_primary_context(device_);
        check(cuInit(0), "cuInit", name_);

        // PTX is SOURCE: the driver's JIT reads it as a C string and runs off the end of a payload
        // that is merely length-prefixed. A cubin is binary and must NOT be terminated.
        if (kind == "cuda_ptx") {
            source_.assign(code.begin(), code.end());
            source_.push_back(0);
            check(cuModuleLoadData(&module_, source_.data()), "cuModuleLoadData (ptx)", name_);
        } else {
            check(cuModuleLoadData(&module_, code.data()), "cuModuleLoadData", name_);
        }
        check(cuModuleGetFunction(&function_, module_, launch_.entry_point.c_str()),
              "cuModuleGetFunction", name_);
    }

    ~KernelStage() override {
        if (module_ != nullptr) {
            cudaSetDevice(device_);
            cuModuleUnload(module_);
        }
    }

    std::string_view get_name() const noexcept override { return name_; }
    deploy::Invocation get_invocation() const noexcept override { return invocation_; }
    const std::vector<std::string>& get_inputs() const noexcept override { return inputs_; }
    const std::vector<std::string>& get_outputs() const noexcept override { return outputs_; }

    std::uint64_t get_elements(std::string_view tensor, bool is_input) const override {
        const auto& names = is_input ? inputs_ : outputs_;
        const auto& counts = is_input ? input_elements_ : output_elements_;
        for (std::size_t i = 0; i < names.size(); ++i)
            if (names[i] == tensor)
                return counts[i] * static_cast<std::uint64_t>(rows_ > 0 ? rows_ : 1);
        throw Exception::not_found(is_input ? "kernel input" : "kernel output", tensor);
    }

    void set_rows(std::int64_t rows) override {
        rows_ = rows;
        bound_.assign(inputs_.size() + outputs_.size(), nullptr);
        rebuild_arguments();
    }

    std::int64_t get_rows() const noexcept override { return rows_; }

    bool is_bound(std::string_view tensor, bool is_input) const override {
        const auto& names = is_input ? inputs_ : outputs_;
        for (std::size_t i = 0; i < names.size(); ++i)
            if (names[i] == tensor) return bound_[is_input ? i : inputs_.size() + i] != nullptr;
        return false;
    }

    void bind(std::string_view tensor, bool is_input, std::shared_ptr<memory::MemoryRef> memory) override {
        if (!memory) throw Exception::invalid_argument("bind: null memory reference");
        // A kernel dereferences the address directly, so memory it cannot reach is a crash rather
        // than a wrong number. Refuse it here, where the domain can still be named.
        memory->require_domain(memory::Domain::Cuda);
        const auto& names = is_input ? inputs_ : outputs_;
        for (std::size_t i = 0; i < names.size(); ++i)
            if (names[i] == tensor) {
                bound_[is_input ? i : inputs_.size() + i] = std::move(memory);
                rebuild_arguments();
                return;
            }
        throw Exception::not_found(is_input ? "kernel input" : "kernel output", tensor);
    }

    /// The stage's weights, bound once and never rebound: they do not change with the grid.
    void set_weights(std::shared_ptr<memory::MemoryRef> weights) override {
        if (weights) weights->require_domain(memory::Domain::Cuda);
        weights_ = std::move(weights);
        rebuild_arguments();
    }

    void run() override {
        for (std::size_t i = 0; i < bound_.size(); ++i)
            if (!bound_[i])
                throw Exception::invalid_argument(
                    "stage `" + name_ + "`: nothing bound for " +
                    (i < inputs_.size() ? "input `" + inputs_[i] + "`"
                                        : "output `" + outputs_[i - inputs_.size()] + "`"));

        // `cuLaunchKernel` launches into the THREAD's current context, not the module's. A function
        // loaded on device 1 and launched while device 0 is current is an invalid handle, so the
        // device is re-established every run rather than assumed to have survived whatever else ran
        // on this thread. On a single-GPU machine this is a no-op, which is exactly why it is easy
        // to leave out and impossible to notice.
        use_primary_context(device_);

        const auto grid = launch_.get_grid_for(static_cast<std::uint64_t>(rows_));
        check(cuLaunchKernel(function_, grid[0], grid[1], grid[2], launch_.block_size[0],
                             launch_.block_size[1], launch_.block_size[2], launch_.shared_memory_bytes,
                             /*stream=*/nullptr, arguments_.data(), nullptr),
              "cuLaunchKernel", name_);
        // The default stream is synchronised against by the next ONNX stage's own work, but a
        // launch error surfaces asynchronously, so a stage that faulted would otherwise be blamed on
        // whatever ran next.
        check(cuCtxSynchronize(), "cuCtxSynchronize", name_);
    }

private:
    static std::uint64_t elements_of(const deploy::Composition& composition, const std::string& buffer,
                                     const std::string& stage) {
        if (const auto* declared = composition.get_buffer(buffer)) return declared->elements;
        throw Exception::invalid_package("stage `" + stage + "` names buffer `" + buffer +
                                         "`, which the composition does not declare");
    }

    /// Lay out the argument vector: weights, then inputs, then outputs, then the row count.
    ///
    /// Rebuilt on a bind rather than per run, so `run()` allocates and computes nothing. The
    /// `CUdeviceptr` values are held as members because `cuLaunchKernel` takes POINTERS TO the
    /// argument values — a vector of addresses into a temporary would dangle by the time it reads.
    void rebuild_arguments() {
        addresses_.assign(1 + bound_.size(), 0);
        addresses_[0] = weights_ ? static_cast<CUdeviceptr>(weights_->get_address() +
                                                            weights_->get_offset_bytes())
                                 : CUdeviceptr{0};
        for (std::size_t i = 0; i < bound_.size(); ++i)
            addresses_[i + 1] = bound_[i] ? static_cast<CUdeviceptr>(bound_[i]->get_address() +
                                                                     bound_[i]->get_offset_bytes())
                                          : CUdeviceptr{0};
        row_count_ = static_cast<std::uint32_t>(rows_);

        arguments_.assign(addresses_.size() + 1, nullptr);
        for (std::size_t i = 0; i < addresses_.size(); ++i) arguments_[i] = &addresses_[i];
        arguments_.back() = &row_count_;
    }

    std::string name_;
    deploy::Invocation invocation_;
    deploy::KernelLaunch launch_;
    int device_;
    std::int64_t rows_ = 0;

    std::vector<std::string> inputs_, outputs_;
    /// Per ROW, from the composition's buffer table.
    std::vector<std::uint64_t> input_elements_, output_elements_;
    /// Inputs first, then outputs — the same order as `arguments_` after the weights.
    std::vector<std::shared_ptr<memory::MemoryRef>> bound_;
    std::shared_ptr<memory::MemoryRef> weights_;

    /// NUL-terminated copy of a PTX payload; empty for a cubin.
    std::vector<std::uint8_t> source_;
    CUmodule module_ = nullptr;
    CUfunction function_ = nullptr;

    std::vector<CUdeviceptr> addresses_;
    std::uint32_t row_count_ = 0;
    std::vector<void*> arguments_;
};

}  // namespace
#endif

std::unique_ptr<Stage> Backend::make_kernel_stage(
    [[maybe_unused]] const deploy::Composition& composition, const deploy::Stage& stage,
    [[maybe_unused]] std::string_view kind, [[maybe_unused]] deploy::byte_view code,
    [[maybe_unused]] const deploy::KernelLaunch& launch, [[maybe_unused]] int device) const {
#ifndef RFNN_WITH_CUDA
    throw Exception::feature_disabled("cuda", "running stage `" + stage.name + "` as compiled code");
#else
    (void)stage;
    return std::make_unique<KernelStage>(composition, stage, kind, code, launch, device);
#endif
}

const ComputeBackend& compute_backend() {
    static const Backend backend;
    return backend;
}

}  // namespace RadFiled3D::nn::cuda
