#include <RadFiled3D/nn/backends/onnx.hpp>

#include <RadFiled3D/nn/backends/compute_backend.hpp>
#include <RadFiled3D/nn/core/stage_weights.hpp>
#include <RadFiled3D/nn/memory/memory_ref.hpp>

#include <onnxruntime_cxx_api.h>
#ifdef RFNN_WITH_DIRECTML
// Only in the DirectML ONNX Runtime package (NuGet), never in the GitHub release archives — which
// is why cmake/OnnxRuntime.cmake fetches a different source for this backend. `d3d12.h` is needed
// for the ID3D12Resource type alone; no Direct3D function is called.
#include <d3d12.h>
#include <dml_provider_factory.h>
#endif

#include <algorithm>
#include <set>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace RadFiled3D::nn::onnx {

// ONNX is inherently active as the whole use of this package is to use ONNX primarily
bool available() noexcept {
    return true;
}

std::optional<std::string> version() {
    const OrtApiBase* base = OrtGetApiBase();
    if (!base) return std::nullopt;
    const char* v = base->GetVersionString();
    return v ? std::optional<std::string>(v) : std::nullopt;
}

namespace {

/// One ORT environment per process, and DELIBERATELY NEVER DESTROYED.
///
/// ORT wants exactly one environment and hangs its thread pools, logger and provider registry off
/// it, so it cannot be a local. It must also not be an ordinary static: destroying it at process
/// exit races the teardown of the CUDA runtime and of the provider shared objects it dlopen'd, and
/// the loser is a heap corruption ("corrupted double-linked list") long after the last test passed.
/// A process-lifetime singleton that is never freed has no such race, and the OS reclaims it anyway.
Ort::Env& environment() {
    static Ort::Env* env = new Ort::Env{ORT_LOGGING_LEVEL_WARNING, "radfiled3d-nn"};
    return *env;
}

/// The memory ORT should expect for a reference's domain.
///
/// This is the whole reason bindings speak `MemoryRef`: a host vector and a buffer imported from
/// Unreal's Vulkan heap differ here and nowhere else in the call path.
Ort::MemoryInfo memory_info_for(const memory::MemoryRef& ref, const ComputeBackend& backend, int device) {
    if (ref.get_domain() == memory::Domain::Host)
        return Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    // The allocator name comes FROM THE BACKEND, so this file names no backend of its own. A
    // reference must be in the running backend's own domain: a VkBuffer or an ID3D12Resource has to
    // have been adopted first (`ComputeBackend::adopt`, which the bind path runs), and memory
    // belonging to a different compute backend is not ours to bind either.
    if (ref.get_domain() != backend.get_memory_domain())
        throw Exception::invalid_argument(
            std::string("cannot bind ") + std::string(memory::to_string(ref.get_domain())) +
            " memory to a " + std::string(to_string(backend.get_backend())) +
            " session; import it into that backend first");
    if (backend.get_ort_allocator_name().empty())
        // The reference IS in this backend's domain — the backend just has no way to hand ORT
        // caller-owned memory yet. Saying "import it first" here would be a lie that sends someone
        // to fix the wrong thing.
        throw Exception::feature_disabled(
            backend.get_feature(),
            "binding caller-owned device memory on this backend (its ONNX Runtime allocator is not wired up)");
    return Ort::MemoryInfo(std::string(backend.get_ort_allocator_name()).c_str(), OrtDeviceAllocator,
                           device, OrtMemTypeDefault);
}

#ifdef RFNN_WITH_DIRECTML
/// ORT's DirectML extension API. Reached through the core API by name — `GetExecutionProviderApi`
/// is the documented way in, and "DML" is the only provider name it accepts.
const OrtDmlApi* dml_api() {
    const OrtDmlApi* api = nullptr;
    Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION,
                                                            reinterpret_cast<const void**>(&api)));
    if (api == nullptr) throw Exception::feature_disabled("directml", "reaching the DirectML provider API");
    return api;
}
#endif

/// The file name an ONNX graph refers to its external initializers by, in ORT's own path type.
///
/// `ORTCHAR_T` is `wchar_t` on Windows and `char` everywhere else; the name is recorded in the
/// package as UTF-8 either way, because the container is one format on every platform. The names
/// exporters write are ASCII file names (`model.onnx.data`), so widening byte by byte is exact.
std::basic_string<ORTCHAR_T> to_ort_path(std::string_view utf8) {
    return std::basic_string<ORTCHAR_T>(utf8.begin(), utf8.end());
}

/// A stage that is an ONNX graph, run through ONNX Runtime's IoBinding.
///
/// One of the two `nn::Stage` implementations — `nn::cuda::KernelStage` is the other — so a session
/// runs a chain of graphs, a chain of kernels, or any mixture without knowing which is which.
///
/// Bindings are registered once per grid and re-read on every run, which is what R-I1 promises: a
/// caller editing its input buffer in place is picked up without rebinding, and `run()` allocates
/// nothing.
class GraphStage final : public nn::Stage {
public:
    GraphStage(std::string name, deploy::Invocation invocation, Ort::Env& env, deploy::byte_view graph,
               const Ort::SessionOptions& options, Backend backend, int device)
        : name_(std::move(name)), invocation_(invocation), backend_(backend), device_(device) {
        session_ = std::make_unique<Ort::Session>(env, graph.data(), graph.size(), options);
        binding_ = std::make_unique<Ort::IoBinding>(*session_);
        Ort::AllocatorWithDefaultOptions allocator;
        for (std::size_t i = 0; i < session_->GetInputCount(); ++i) {
            inputs_.emplace_back(session_->GetInputNameAllocated(i, allocator).get());
            input_shapes_.push_back(session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }
        for (std::size_t i = 0; i < session_->GetOutputCount(); ++i) {
            outputs_.emplace_back(session_->GetOutputNameAllocated(i, allocator).get());
            output_shapes_.push_back(session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }
    }

    std::string_view get_name() const noexcept override { return name_; }
    deploy::Invocation get_invocation() const noexcept override { return invocation_; }
    const std::vector<std::string>& get_inputs() const noexcept override { return inputs_; }
    const std::vector<std::string>& get_outputs() const noexcept override { return outputs_; }
    std::int64_t get_rows() const noexcept override { return rows_; }

    std::uint64_t get_elements(std::string_view tensor, bool is_input) const override {
        const auto shape = get_shape(tensor, is_input);
        return static_cast<std::uint64_t>(
            std::accumulate(shape.begin(), shape.end(), std::int64_t{1}, std::multiplies<>()));
    }

    /// The graph's own shape for a tensor, with every dynamic axis resolved to the row count.
    ///
    /// From the GRAPH, not from the descriptor: the two disagree in practice (a `flux` output is
    /// rank 1 where the descriptor says shape {1}), and only one of them is what ORT validates.
    std::vector<std::int64_t> get_shape(std::string_view tensor, bool is_input) const {
        const auto& names = is_input ? inputs_ : outputs_;
        const auto& shapes = is_input ? input_shapes_ : output_shapes_;
        const auto found = std::find(names.begin(), names.end(), tensor);
        if (found == names.end())
            throw Exception::not_found(is_input ? "graph input" : "graph output", tensor);
        std::vector<std::int64_t> shape = shapes[static_cast<std::size_t>(found - names.begin())];
        for (auto& dim : shape)
            if (dim < 0) dim = rows_;
        return shape;
    }

    void set_rows(std::int64_t rows) override {
        rows_ = rows;
        // The bound shapes depend on the row count, so a grid change invalidates them rather than
        // silently running the previous size.
        binding_->ClearBoundInputs();
        binding_->ClearBoundOutputs();
        bound_inputs_.clear();
        bound_outputs_.clear();
    }

    bool is_bound(std::string_view tensor, bool is_input) const override {
        const auto& bound = is_input ? bound_inputs_ : bound_outputs_;
        return bound.find(std::string(tensor)) != bound.end();
    }

    void bind(std::string_view tensor, bool is_input, std::shared_ptr<memory::MemoryRef> memory) override {
        if (!memory) throw Exception::invalid_argument("bind: null memory reference");
        const std::string name(tensor);
        const auto shape = get_shape(name, is_input);
        const auto elements = static_cast<std::uint64_t>(
            std::accumulate(shape.begin(), shape.end(), std::int64_t{1}, std::multiplies<>()));
        // Capacity is checked in ELEMENTS, never bytes (R-I1) — a half-sized buffer of the right
        // byte count is still the wrong buffer. A LARGER one is fine and is how a producer writes
        // the leading rows of an allocation its readers read whole.
        const std::uint64_t capacity = memory->get_element_count(sizeof(float));
        if (capacity < elements)
            throw Exception::invalid_argument("binding `" + name + "` needs " + std::to_string(elements) +
                                              " float elements, the buffer holds " +
                                              std::to_string(capacity));

        const ComputeBackend& backend = get_compute_backend(backend_);
        // The reference is held for the lifetime of the binding: ORT keeps the raw pointer, so
        // whoever owns the memory has to stay alive and this is where that is guaranteed.
        if (is_input) {
            binding_->BindInput(name.c_str(), tensor_for(*memory, shape, elements, backend));
            bound_inputs_.insert_or_assign(name, std::move(memory));
        } else {
            binding_->BindOutput(name.c_str(), tensor_for(*memory, shape, elements, backend));
            bound_outputs_.insert_or_assign(name, std::move(memory));
        }
    }

    Ort::Value tensor_for(const memory::MemoryRef& memory, const std::vector<std::int64_t>& shape,
                          std::uint64_t elements, const ComputeBackend& backend) {
        Ort::MemoryInfo info = memory_info_for(memory, backend, device_);
        void* address = reinterpret_cast<void*>(
            static_cast<std::uintptr_t>(memory.get_address() + memory.get_offset_bytes()));
        return Ort::Value::CreateTensor(info, address, elements * sizeof(float), shape.data(),
                                        shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    }

    void run() override {
        session_->Run(Ort::RunOptions{nullptr}, *binding_);
        binding_->SynchronizeOutputs();
    }

private:
    std::string name_;
    deploy::Invocation invocation_;
    Backend backend_;
    int device_;
    std::int64_t rows_ = 0;

    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::IoBinding> binding_;
    std::vector<std::string> inputs_, outputs_;
    std::vector<std::vector<std::int64_t>> input_shapes_, output_shapes_;
    std::map<std::string, std::shared_ptr<memory::MemoryRef>> bound_inputs_, bound_outputs_;
};

/// One of the composition's named buffers, allocated for the life of a grid.
///
/// A stage writes into it and ANY later stage reads from it — that is what makes a value computed
/// once reach every stage that needs it without passing through the caller. Allocated at
/// `set_voxel_grid` and not touched again until the grid changes, so a run only executes: the
/// pointers ORT and the driver were handed stay valid across every inference.
///
/// It lives in the BACKEND's own memory domain — device memory on CUDA — so a kernel stage can read
/// it directly and ONNX Runtime binds it without a host round trip. That is what makes a mixed chain
/// free of copies: nothing crosses the bus between stages.
///
/// Sized for the LARGEST reader, with the producer writing its leading part, so a value computed
/// once and needed per query is served by repeating that part in place — no second allocation and no
/// copy beyond the repeat itself.
///
/// The relationship is decided by ELEMENT COUNTS, not by a batch axis, because not every carried
/// tensor has one. `beam_encoder.latent` is `[1, 192]` into the trunk's `[queries, 192]` and repeats;
/// `encoding_config.region_state` is `[14]` into the trunk's `[14]` and does not, even though the
/// trunk itself runs per query. A rule phrased in rows would get the second one wrong.
struct SharedBuffer {
    std::string name;
    /// The stage that fills it.
    std::string writer;
    std::shared_ptr<memory::MemoryRef> memory;
    /// What one run of the writer puts in, in elements.
    std::int64_t produced = 0;
    /// How many times that block fills the largest reader. 1 means "carried as-is".
    std::int64_t repeats = 1;

    /// Replicate the producer's single row across the batch, in place and in the buffer's own domain.
    ///
    /// By DOUBLING the part already written, not by copying the first row once per repeat: the
    /// copies are device-to-device and tiny (a 192-wide latent is 768 bytes), so a per-row loop pays
    /// the per-call overhead a quarter of a million times for a 64³ grid and costs more than the
    /// network it feeds — measured at 195 ms of a 201 ms frame. Doubling needs ~log2(repeats) calls,
    /// and each one moves twice what the last did. Source and destination never overlap: the block
    /// copied is at most as long as the prefix already filled, so it ends where the destination
    /// begins.
    void repeat(const ComputeBackend& backend) const {
        if (repeats <= 1 || produced <= 0) return;
        const auto span = static_cast<std::uint64_t>(produced) * sizeof(float);
        for (std::int64_t filled = 1; filled < repeats;) {
            const std::int64_t chunk = std::min(filled, repeats - filled);
            backend.copy_within(*memory, static_cast<std::uint64_t>(filled) * span, *memory, 0,
                                static_cast<std::uint64_t>(chunk) * span);
            filled += chunk;
        }
    }
};

/// A session over a package's graphs, with its I/O bound from caller-owned memory.
///
/// A package that records a composition (`deploy::Composition`) is driven as the wiring says: the
/// stages run in the recorded order, and a tensor one stage produces is carried to the input of a
/// later one without ever reaching the caller. A package that records none is a single `trunk`,
/// driven directly — the behaviour every package had before the format could say otherwise.
class OrtSession final : public InferenceSession {
public:
    OrtSession(deploy::Package package, Backend backend, int device)
        : model_(std::move(package)), backend_(backend), compute_(get_compute_backend(backend)) {
        // Resolve the device ONCE, here, and use it for everything after: the execution provider,
        // the intermediate buffers, a kernel stage's module and its launches, and the weights
        // uploaded for it. A negative ordinal means "whatever the backend's current device is",
        // which is device 0 unless the process already chose otherwise.
        device_ = device < 0 ? 0 : device;
        // The environment FIRST, before any other ORT object exists. Appending an execution provider
        // logs, and ORT logs through the environment's logger — construct one later and the provider
        // aborts with "Attempt to use DefaultLogger but none has been registered".
        Ort::Env& env = environment();

        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // BEFORE the device is validated: a backend that was never compiled in has no devices to
        // count, and "cuda has no devices" would send someone hunting for a driver problem when the
        // answer is a CMake option. `append_provider` is what says that, so it goes first.
        append_provider(options);

        // Only an EXPLICIT ordinal is checked. `automatic()` is a request to use whatever is there,
        // so it must not turn a machine with no usable card into an error the caller did not ask
        // about — the provider above already refused that case with a better message.
        if (device >= 0) {
            const int count = compute_.get_device_count();
            if (device >= count)
                throw Exception::invalid_argument(
                    "device " + std::to_string(device) + " was asked for, but " +
                    std::string(to_string(backend)) +
                    (count == 0 ? std::string(" has no devices")
                                : " has " + std::to_string(count) +
                                      (count == 1 ? " device" : " devices")));
        }

        // The composition names what runs and in what order. `Model` resolved it, along with the
        // per-stage weights — including the bytes an ONNX graph's external initializers point at,
        // which must be alive while the session is CONSTRUCTED and stay put for its whole life. A
        // grid change does not alter what a model weighs.
        //
        // Without a composition there is one stage, so the rest of this class needs no second code
        // path for the single-graph case.
        if (model_.get_composition())
            plan_ = model_.get_composition()->stages;
        else
            plan_.push_back({std::string(deploy::kTrunkGraph), deploy::Invocation::PerQuery, {}, {}, {}, {}});

        // Before anything is compiled: does this machine carry code for every stage at all? A
        // package may legitimately ship a stage only as `cuda_ptx`, and a user on the wrong card is
        // owed the stage name and the architecture rather than an ORT error about a missing graph.
        arch_ = compute_.get_device_arch(device_);
        // The backend's HARDWARE FAMILY, not its name: TensorRT and CUDA are two providers on one
        // card, and a `cuda_ptx` block runs under both.
        target_ = compute_.get_block_target();
        model_.get_package().require_runnable_on(target_, arch_, to_string(backend_));

        for (const auto& declared : plan_) stages_.push_back(build_stage(declared, env, options));
    }

    ~OrtSession() override { release_dml_allocations(); }

    const deploy::Package& get_package() const noexcept override { return model_.get_package(); }
    const Model& get_model() const noexcept override { return model_; }

    const StageWeights& get_stage_weights(std::string_view stage) const override {
        return model_.get_stage_weights(stage);
    }

    Backend get_backend() const noexcept override { return backend_; }
    int get_device() const noexcept override { return device_; }

    void set_voxel_grid(std::array<std::uint32_t, 3> voxel_counts) override {
        for (const auto c : voxel_counts)
            if (c == 0) throw Exception::invalid_argument("voxel counts must all be positive");
        queries_ = static_cast<std::int64_t>(voxel_counts[0]) * voxel_counts[1] * voxel_counts[2];
        release_dml_allocations();
        for (std::size_t i = 0; i < stages_.size(); ++i)
            stages_[i]->set_rows(Stage::rows_for(plan_[i].invocation, queries_));
        allocate_buffers();
    }

    void bind_input(std::string_view name, std::shared_ptr<memory::MemoryRef> buffer) override {
        bind(name, std::move(buffer), /*is_input=*/true);
    }

    void bind_output(std::string_view name, std::shared_ptr<memory::MemoryRef> buffer) override {
        bind(name, std::move(buffer), /*is_input=*/false);
    }

    void infer() override {
        if (queries_ == 0)
            throw Exception::invalid_argument("set_voxel_grid must be called before infer()");

        // Report EVERY missing binding at once (R-I1): finding them one run at a time is the thing
        // that makes driving a model tedious. A tensor a buffer feeds is NOT missing — it is the
        // composition's to supply, and asking the caller for it would be asking for the very thing
        // the wiring exists to remove.
        std::vector<std::string> missing;
        for (std::size_t i = 0; i < stages_.size(); ++i) {
            const Stage& stage = *stages_[i];
            for (const auto& name : stage.get_inputs())
                if (!get_read(plan_[i], name) && !stage.is_bound(name, true))
                    missing.push_back("input `" + name + "`" + stage_suffix(stage));
            for (const auto& name : stage.get_outputs())
                if (!get_write(plan_[i], name) && !stage.is_bound(name, false))
                    missing.push_back("output `" + name + "`" + stage_suffix(stage));
        }
        if (!missing.empty()) {
            std::string all;
            for (const auto& m : missing) all += (all.empty() ? "" : ", ") + m;
            throw Exception::invalid_argument("infer(): nothing bound for " + all);
        }

        // In the recorded order, so a stage's inputs are computed before it reads them. Everything
        // was bound once, at set_voxel_grid, and points at memory this session owns and does not
        // move — so a run executes and replicates, and allocates nothing.
        for (const auto& stage : stages_) {
            // A skipped stage is simply not run. Its output buffer still holds what the last run
            // put there and every reader downstream sees that, which is the whole mechanism: a
            // global state encoded once is reused by leaving it alone, not by copying it forward.
            if (disabled_.contains(stage->get_name())) continue;
            stage->run();
            // A `Once` producer wrote one row; its consumers want one per query.
            for (const auto& buffer : buffers_)
                if (buffer.writer == stage->get_name()) buffer.repeat(compute_);
        }
    }

    const std::shared_ptr<memory::MemoryRef>& get_stage_buffer(std::string_view name) const override {
        for (const auto& buffer : buffers_)
            if (buffer.name == name) return buffer.memory;
        throw Exception::not_found("stage buffer", name);
    }

    std::vector<std::string> get_stage_buffers() const override {
        std::vector<std::string> names;
        names.reserve(buffers_.size());
        for (const auto& buffer : buffers_) names.push_back(buffer.name);
        return names;
    }

    void set_stage_enabled(std::string_view stage, bool enabled) override {
        // Checked against the PLAN, so a typo is refused rather than silently doing nothing — a
        // skip that quietly failed would show up as a stale result nobody could explain.
        const bool known = std::any_of(plan_.begin(), plan_.end(),
                                       [&](const deploy::Stage& s) { return s.name == stage; });
        if (!known) throw Exception::not_found("stage", stage);
        if (enabled)
            disabled_.erase(std::string(stage));
        else
            disabled_.insert(std::string(stage));
    }

    bool is_stage_enabled(std::string_view stage) const override {
        const bool known = std::any_of(plan_.begin(), plan_.end(),
                                       [&](const deploy::Stage& s) { return s.name == stage; });
        if (!known) throw Exception::not_found("stage", stage);
        return !disabled_.contains(std::string(stage));
    }

private:
    void release_dml_allocations() noexcept {
#ifdef RFNN_WITH_DIRECTML
        if (dml_allocations_.empty()) return;
        try {
            const OrtDmlApi* dml = dml_api();
            for (void* allocation : dml_allocations_) dml->FreeGPUAllocation(allocation);
        } catch (...) {
            // A destructor has nowhere to report this, and the process is not harmed: the wrappers
            // go with the session, and the caller's resources were never ours.
        }
        dml_allocations_.clear();
#endif
    }

    void append_provider(Ort::SessionOptions& options) {
        switch (backend_) {
            case Backend::Cpu:
                return;  // the default provider; nothing to append
            case Backend::TensorRt: {
#ifndef RFNN_WITH_TENSORRT
                throw Exception::feature_disabled("tensorrt", "building an inference session");
#else
                // TensorRT first so it claims what it can, CUDA behind it for everything the TRT
                // engine will not take. Appending only TRT would leave unsupported nodes on the CPU
                // and copy across the bus every run.
                //
                // The options are set EXPLICITLY. A zero-initialised struct is not "defaults": ORT
                // reads the zeros as invalid, warns, and substitutes its own — so the build silently
                // depends on ORT's correction rather than on anything stated here.
                OrtTensorRTProviderOptions trt{};
                trt.device_id = device_;
                trt.trt_max_partition_iterations = 1000;
                trt.trt_min_subgraph_size = 1;
                trt.trt_max_workspace_size = std::size_t{1} << 30;  // 1 GiB for engine building
                // fp16 stays OFF. It is a speed/accuracy trade the caller makes deliberately
                // (R-I3), never one this library applies behind their back.
                trt.trt_fp16_enable = 0;
                options.AppendExecutionProvider_TensorRT(trt);
                OrtCUDAProviderOptions cuda{};
                cuda.device_id = device_;
                options.AppendExecutionProvider_CUDA(cuda);
                return;
#endif
            }
            case Backend::Cuda: {
#ifndef RFNN_WITH_CUDA
                throw Exception::feature_disabled("cuda", "building an inference session");
#else
                OrtCUDAProviderOptions cuda{};
                cuda.device_id = device_;
                options.AppendExecutionProvider_CUDA(cuda);
                return;
#endif
            }
            case Backend::Rocm:
                throw Exception::feature_disabled("rocm", "building an inference session");
            case Backend::DirectMl: {
#ifndef RFNN_WITH_DIRECTML
                throw Exception::feature_disabled("directml", "building an inference session");
#else
                // DirectML is appended through its own provider API rather than a struct of options.
                // It also requires the default allocator arena OFF: the EP manages D3D12 memory
                // itself and ORT's arena would allocate on top of it.
                options.DisableMemPattern();
                options.SetExecutionMode(ORT_SEQUENTIAL);
                const OrtDmlApi* dml = dml_api();
                Ort::ThrowOnError(dml->SessionOptionsAppendExecutionProvider_DML(options, device_));
                return;
#endif
            }
        }
        throw Exception::invalid_argument("unknown backend");
    }

    /// Build one stage: the graph if the package has one for it, otherwise the compiled form this
    /// device can run.
    ///
    /// The ONNX graph wins when both exist. It is the portable path and it is what every shipped
    /// package exercises, so a package carrying both means "here is a faster route" — taking it
    /// silently would change results on some machines and not others. A kernel runs when it is the
    /// ONLY form for that stage, which is exactly the case it was added for.
    std::unique_ptr<Stage> build_stage(const deploy::Stage& declared, Ort::Env& env,
                                       Ort::SessionOptions& options) {
        if (const auto graph = model_.get_package().get_graph(declared.name)) {
            // One SessionOptions serves every stage — the provider list is a property of the device
            // this session runs on, not of the individual graph — EXCEPT for a graph that keeps its
            // initializers in an external file. That is per graph, so such a stage gets a clone with
            // the package's own bytes standing in for the file the graph names. Nothing has to sit
            // on disk beside the `.rf3m`, which is the point of a self-contained container.
            const StageWeights& weights = get_stage_weights(declared.name);
            if (weights.get_onnx_external_file().empty() || weights.is_empty())
                return std::make_unique<GraphStage>(declared.name, declared.invocation, env, *graph,
                                                    options, backend_, device_);
            Ort::SessionOptions external = options.Clone();
            const std::vector<std::basic_string<ORTCHAR_T>> names{
                to_ort_path(weights.get_onnx_external_file())};
            const std::vector<char*> buffers{
                const_cast<char*>(reinterpret_cast<const char*>(weights.get_bytes().data()))};
            const std::vector<std::size_t> lengths{weights.get_bytes().size()};
            external.AddExternalInitializersFromFilesInMemory(names, buffers, lengths);
            return std::make_unique<GraphStage>(declared.name, declared.invocation, env, *graph, external,
                                                backend_, device_);
        }

        // `require_runnable_on` passed, so the package DOES carry code for this stage on this
        // device. Pick the most specific form — `select_block` prefers an exact architecture match
        // over a portable one — and take the launch descriptor for THAT form, not for whichever the
        // stage happens to list first: a cubin and the PTX beside it need not share an entry point.
        const deploy::Block* block = model_.get_package().select_block(declared.name, target_, arch_);
        if (block == nullptr)
            throw Exception::invalid_package("stage `" + declared.name +
                                             "` has no form this device can run");
        const deploy::KernelLaunch* launch =
            declared.get_launch_for(block->kind.get_name(), block->target_arch);
        if (launch == nullptr)
            throw Exception::invalid_package(
                "stage `" + declared.name + "` selects its " + std::string(block->kind.get_name()) +
                " form, but declares no launch descriptor for it; a kernel cannot be called without "
                "an entry point");
        if (!model_.get_composition())
            throw Exception::invalid_package("a stage that is compiled code needs a composition to "
                                             "say what it reads and writes");
        std::unique_ptr<Stage> stage =
            compute_.make_kernel_stage(*model_.get_composition(), declared, block->kind.get_name(),
                                       deploy::byte_view(block->payload), *launch, device_);
        // The weights reach a kernel as its FIRST argument (docs/custom-code.md §4), and they must be
        // in the backend's own memory domain — a kernel dereferences the address, it cannot read a
        // host pointer. Uploaded once, here, and never again: a grid does not change what a model
        // weighs.
        const StageWeights& weights = get_stage_weights(declared.name);
        std::shared_ptr<memory::MemoryRef> resident =
            compute_.allocate(weights.get_bytes().size(), device_);
        if (!weights.is_empty())
            compute_.upload(*resident, weights.get_bytes().data(), weights.get_bytes().size());
        kernel_weights_.push_back(resident);
        stage->set_weights(std::move(resident));
        return stage;
    }

    static const deploy::Port* get_read(const deploy::Stage& stage, const std::string& tensor) noexcept {
        for (const auto& p : stage.reads)
            if (p.tensor == tensor) return &p;
        return nullptr;
    }

    static const deploy::Port* get_write(const deploy::Stage& stage, const std::string& tensor) noexcept {
        for (const auto& p : stage.writes)
            if (p.tensor == tensor) return &p;
        return nullptr;
    }

    /// Name the stage in a diagnostic only when there is more than one, so a single-stage package's
    /// messages read exactly as they always did.
    std::string stage_suffix(const Stage& stage) const {
        return stages_.size() > 1 ? " of `" + std::string(stage.get_name()) + "`" : std::string();
    }

    /// The stage that declares `name` on the given side, for a caller that binds by tensor name
    /// alone. A name two stages share is refused rather than resolved by position: a caller that
    /// meant the other one would get no error and the wrong buffer.
    std::size_t stage_declaring(const std::string& name, bool is_input) const {
        std::size_t found = stages_.size();
        for (std::size_t i = 0; i < stages_.size(); ++i) {
            const auto& names = is_input ? stages_[i]->get_inputs() : stages_[i]->get_outputs();
            if (std::find(names.begin(), names.end(), name) == names.end()) continue;
            // A tensor attached to a buffer is internal plumbing; not a candidate for a caller's bind.
            if (is_input && get_read(plan_[i], name)) continue;
            if (!is_input && get_write(plan_[i], name)) continue;
            if (found != stages_.size())
                throw Exception::invalid_argument("`" + name + "` is a " +
                                                  (is_input ? "input" : "output") + " of both `" +
                                                  std::string(stages_[found]->get_name()) + "` and `" +
                                                  std::string(stages_[i]->get_name()) +
                                                  "`; the binding would be ambiguous");
            found = i;
        }
        if (found != stages_.size()) return found;
        // Distinguish "no such tensor" from "that one is the composition's to supply", because the
        // fixes are completely different.
        for (std::size_t i = 0; i < stages_.size(); ++i) {
            const auto& names = is_input ? stages_[i]->get_inputs() : stages_[i]->get_outputs();
            if (std::find(names.begin(), names.end(), name) == names.end()) continue;
            const deploy::Port* port = is_input ? get_read(plan_[i], name) : get_write(plan_[i], name);
            if (port == nullptr) continue;
            // Name the stage at the other end: a caller told only "buffer `latent`" still has to go
            // looking for who fills it.
            std::string other;
            for (std::size_t j = 0; j < stages_.size(); ++j)
                for (const auto& p : (is_input ? plan_[j].writes : plan_[j].reads))
                    if (p.buffer == port->buffer) other = plan_[j].name + "." + p.tensor;
            throw Exception::invalid_argument(
                "`" + name + "` of `" + std::string(stages_[i]->get_name()) + "` is " +
                (is_input ? "fed from" : "written into") + " the composition's buffer `" + port->buffer +
                "`" + (other.empty() ? "" : ", which `" + other + (is_input ? "` writes" : "` reads")) +
                "; the package composes its stages, so it is not yours to bind");
        }
        throw Exception::not_found(is_input ? "graph input" : "graph output", name);
    }

    /// Allocate every named buffer the composition declares, and bind both ends of each.
    ///
    /// Done once per grid rather than per run, which is the whole point of naming them: the memory
    /// belongs to this session, does not move, and stays valid for as long as the model is loaded at
    /// this resolution. A run then only executes and replicates.
    void allocate_buffers() {
        buffers_.clear();
        if (!model_.get_composition()) return;

        const auto find_stage = [this](std::string_view name) -> std::size_t {
            for (std::size_t i = 0; i < plan_.size(); ++i)
                if (plan_[i].name == name) return i;
            return plan_.size();
        };

        for (const auto& declared : model_.get_composition()->buffers) {
            SharedBuffer buffer;
            buffer.name = declared.name;

            // `Composition::validate` proved there is exactly one writer and that every reader runs
            // after it, so the only thing left to check is whether the STAGES agree with what the
            // package declared — which validation cannot see, because reading a package must not
            // require a runtime (rule 3).
            const deploy::Stage* writer = model_.get_composition()->get_writer_of(declared.name);
            const std::size_t producer = writer ? find_stage(writer->name) : plan_.size();
            if (producer == plan_.size())
                throw Exception::invalid_package("buffer `" + declared.name +
                                                 "` has no producing stage in this session");
            buffer.writer = plan_[producer].name;
            std::string written_tensor;
            for (const auto& p : plan_[producer].writes)
                if (p.buffer == declared.name) written_tensor = p.tensor;
            buffer.produced =
                static_cast<std::int64_t>(stages_[producer]->get_elements(written_tensor, false));

            // The declaration is per ROW; the writer emits one row per invocation. A disagreement
            // means the package describes a different model than the code it ships, which is worth
            // saying here rather than letting ORT fail on a shape nobody declared.
            const std::int64_t expected =
                static_cast<std::int64_t>(declared.elements) * stages_[producer]->get_rows();
            if (buffer.produced != expected)
                throw Exception::invalid_package(
                    "buffer `" + declared.name + "` declares " + std::to_string(declared.elements) +
                    " elements per row and `" + plan_[producer].name + "` runs " +
                    std::to_string(stages_[producer]->get_rows()) + " row(s), so " +
                    std::to_string(expected) + " elements were expected; its `" + written_tensor +
                    "` produces " + std::to_string(buffer.produced));

            // Sized for the LARGEST reader: one reading a single row reads the leading part, one
            // reading the whole batch reads the repeated region, and both share the allocation.
            std::int64_t widest = buffer.produced;
            for (std::size_t i = 0; i < stages_.size(); ++i)
                for (const auto& port : plan_[i].reads) {
                    if (port.buffer != declared.name) continue;
                    const auto consumed =
                        static_cast<std::int64_t>(stages_[i]->get_elements(port.tensor, true));
                    // Either the producer fills the consumer exactly, or it fills a whole number of
                    // copies of it. Anything else is a composition that cannot be honoured, and
                    // saying so here — naming both ends — beats a shape error naming neither.
                    if (buffer.produced <= 0 || consumed % buffer.produced != 0)
                        throw Exception::invalid_package(
                            "buffer `" + declared.name + "` carries " + std::to_string(buffer.produced) +
                            " elements from `" + plan_[producer].name + "` into `" + plan_[i].name + "." +
                            port.tensor + "` (" + std::to_string(consumed) +
                            "); the first does not fit a whole number of times into the second");
                    if (consumed > widest) widest = consumed;
                }
            buffer.repeats = widest / buffer.produced;
            if (buffer.repeats > 1 && plan_[producer].invocation != deploy::Invocation::Once)
                throw Exception::invalid_package(
                    "buffer `" + declared.name + "` would have to repeat what `" + plan_[producer].name +
                    "` writes, but that stage runs per query; only a value computed once may be "
                    "reused across queries");

            // In the BACKEND's domain, so a kernel reads it directly and ORT binds it without a host
            // round trip. This is the allocation that lives as long as the grid does.
            buffer.memory = compute_.allocate(static_cast<std::uint64_t>(widest) * sizeof(float), device_);
            buffers_.push_back(std::move(buffer));
        }

        for (const auto& buffer : buffers_) {
            const std::size_t producer = find_stage(buffer.writer);
            for (const auto& p : plan_[producer].writes)
                if (p.buffer == buffer.name)
                    // The producer writes the FIRST rows of the same allocation its readers read
                    // whole: its own shape is smaller, so binding the buffer takes only that part.
                    stages_[producer]->bind(p.tensor, false, buffer.memory);
            for (std::size_t i = 0; i < stages_.size(); ++i)
                for (const auto& port : plan_[i].reads)
                    if (port.buffer == buffer.name) stages_[i]->bind(port.tensor, true, buffer.memory);
        }
    }

    void bind(std::string_view name_view, std::shared_ptr<memory::MemoryRef> buffer, bool is_input) {
        if (!buffer) throw Exception::invalid_argument("bind: null memory reference");
        if (queries_ == 0)
            throw Exception::invalid_argument("set_voxel_grid must be called before binding");
        // Memory from ANOTHER device is a pointer that means nothing in this session's context —
        // the driver would fault, or worse, read whatever lives at that address on this card. The
        // documentation used to say the two "must" match; this is what makes it so. A reference that
        // does not know its device (host memory, or a bare pointer) is not refused: -1 is "no such
        // notion", never "device 0" (see `MemoryRef::get_device_index`).
        if (const int on = buffer->get_device_index(); on >= 0 && on != device_)
            throw Exception::invalid_argument(
                "binding `" + std::string(name_view) + "`: the buffer is on " +
                std::string(memory::to_string(buffer->get_domain())) + " device " + std::to_string(on) +
                ", this session runs on device " + std::to_string(device_) +
                "; load the session with Device::of(buffer) so both land on the same card");
        // Whatever the caller is holding, make it this backend's. Already ours -> handed straight
        // back. Memory a renderer owns that was imported once -> imported here too, giving a second
        // view of the SAME allocation rather than a copy, and asking twice returns the same
        // reference. Anything with no path -> refused here, naming what would work.
        buffer = compute_.adopt(std::move(buffer), device_);
        const std::string name(name_view);
        stages_[stage_declaring(name, is_input)]->bind(name, is_input, std::move(buffer));
    }

    Model model_;
    /// Stages the caller has switched off. Names, not indices: a plan is rebuilt when the grid
    /// changes and an index would then point at a different stage.
    std::set<std::string, std::less<>> disabled_;
    Backend backend_;
    int device_ = 0;
    const ComputeBackend& compute_;
    /// The device architecture this session runs on (`sm_86`) and the hardware family its
    /// specialised blocks are keyed by (`cuda`), both as the backend reports them. Together they
    /// decide which block and which launch descriptor are selected.
    std::string arch_;
    std::string target_;
    std::int64_t queries_ = 0;
    /// The wiring the package records, or nothing for a single-stage model.
    /// What the composition says, in execution order — the ports and invocations `stages_` do not
    /// carry. Index-aligned with `stages_`.
    std::vector<deploy::Stage> plan_;
    /// The executors, in execution order. A graph or a kernel; this class never asks which.
    std::vector<std::unique_ptr<Stage>> stages_;
    /// One per stage, in the same order. Built once and never rebuilt: a grid chooses how many
    /// queries run, not what the model weighs.
    /// Device-resident copies of the weights a KERNEL stage reads, held for the session's life
    /// because the kernel holds their address.
    std::vector<std::shared_ptr<memory::MemoryRef>> kernel_weights_;
    /// The composition's named intermediates, one allocation each in the backend's own memory
    /// domain, instantiated when a grid is chosen and kept valid for as long as the model is loaded
    /// at that resolution. Rebuilt only on a grid change, since their size depends on the query count.
    std::vector<SharedBuffer> buffers_;
#ifdef RFNN_WITH_DIRECTML
    /// DML allocations made for bound D3D12 resources. They wrap memory this session does not own,
    /// so freeing one releases the wrapper and never the caller's resource.
    std::vector<void*> dml_allocations_;
#endif
};

}  // namespace

std::unique_ptr<InferenceSession> load([[maybe_unused]] deploy::Package package,
                                       [[maybe_unused]] Backend backend, [[maybe_unused]] int device) {
    return std::make_unique<OrtSession>(std::move(package), backend, device);
}

}  
