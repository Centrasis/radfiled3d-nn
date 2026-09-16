// Execution backends and the binding protocol.
//
// The protocol is stated in requirements.md R-I1 and is the reason the runtime exists at all:
//
//     set_voxel_grid({D,H,W}) → bind_input("beam_direction", buf) → bind_output("flux", buf) → infer()
//
// The caller owns every buffer and the session allocates no result. Bindings are registered once
// and re-read on every run, so editing an input in place is picked up without rebinding.
//
// Every backend is an ONNX Runtime execution provider, so `load` is one dispatch and which provider
// gets appended is the backend's own business (`src/backends/onnx.cpp`). A backend compiled out
// fails with a sentence naming the feature rather than a link error, because the full API surface
// exists either way (rule 6).
//
// Values crossing this interface are in GRAPH UNITS. The session never applies a normalizer — see
// `bind_input` for why it cannot — so a caller converts, or drives the model through
// `RadFiled3D/nn/core/field_inference.hpp`, which converts both ways.
#pragma once

#include <RadFiled3D/nn/core/stage_weights.hpp>
#include <RadFiled3D/nn/deploy/package.hpp>
#include <RadFiled3D/nn/exception.hpp>
#include <RadFiled3D/nn/memory/memory_ref.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>

namespace RadFiled3D::nn {

/// Which execution provider runs the graphs.
///
/// The vocabulary is defined HERE, ONCE. Python and the C ABI forward a name into
/// `backend_from_name` and re-decide nothing (R-I3) — two places that decide what "cuda" means are
/// two places that can disagree.
enum class Backend : std::uint8_t {
    /// The portable CPU execution provider.
    Cpu,
    /// NVIDIA, via the CUDA execution provider.
    Cuda,
    /// NVIDIA, via the TensorRT execution provider.
    TensorRt,
    /// AMD, via the ROCm / MIGraphX execution provider.
    Rocm,
    /// Windows, vendor-neutral, via the DirectML execution provider. Runs on D3D12, which is what
    /// makes an engine-owned D3D12 resource its *native* memory rather than an import.
    DirectMl,
};

inline constexpr std::array<Backend, 5> kBackends = {Backend::Cpu, Backend::Cuda, Backend::TensorRt,
                                                     Backend::Rocm, Backend::DirectMl};

std::string_view to_string(Backend backend) noexcept;
/// Throws `InvalidArgument` naming the accepted vocabulary.
Backend backend_from_name(std::string_view name);

/// The CMake option / feature this backend needs. Even `Cpu` needs one: the CPU provider is part of
/// the ONNX Runtime, which is fetched only with `onnx`.
std::string_view get_required_feature(Backend backend) noexcept;
/// Whether this build can actually run the backend.
bool is_available(Backend backend) noexcept;
/// Whether this backend can import memory allocated by a graphics API. The CPU provider cannot;
/// every device provider can, through its own external-memory mechanism (CUDA's
/// `cudaImportExternalMemory`, HIP's equivalent, or — for DirectML — no import at all, because a
/// D3D12 resource already *is* its memory).
bool can_share_graphics_memory(Backend backend) noexcept;

/// Which GPU a session runs on.
///
/// Three ways to say it, and the middle one is what a renderer uses:
///
///     Device::automatic()                 whatever the backend's current device is
///     Device::of(*imported)               the device that memory lives on
///     Device::ordinal(1)                  an explicit index
///
/// **The graphics flow is: import first, then take the device from what you imported.** A renderer
/// already decided which GPU its buffer lives on; `ExternalBuffer::device_uuid` carries that across
/// and `import_external_memory` resolves it to an ordinal. Passing the imported reference here makes
/// the session land on the same card, which is the one arrangement that works — the alternative is a
/// session on device 0 handed a pointer that only means something in device 1's context.
///
/// Once chosen, that device is used for EVERYTHING: the execution provider, the intermediate
/// buffers, a kernel stage's module and its launches, and the weights uploaded for it. A session
/// reports it through `get_device()` and refuses memory from another one.
class Device {
public:
    /// The backend's own current device — device 0 unless something already chose otherwise.
    static Device automatic() noexcept { return Device(-1); }

    /// An explicit ordinal. Checked against the backend's device count when the session is built,
    /// not here: how many devices exist is the backend's business and this type names no backend.
    static Device ordinal(int index) noexcept { return Device(index); }

    /// The device a reference's memory lives on.
    ///
    /// A reference that does not know — host memory, which binds to a session on any device, or one
    /// built from a bare pointer nobody annotated — yields `automatic()`. That is the honest answer:
    /// inventing device 0 would silently undo the choice a caller was trying to make.
    static Device of(const memory::MemoryRef& memory) noexcept {
        return Device(memory.get_device_index());
    }

    /// -1 when automatic.
    int get_index() const noexcept { return index_; }
    bool is_automatic() const noexcept { return index_ < 0; }

    bool operator==(const Device&) const = default;

private:
    explicit Device(int index) noexcept : index_(index) {}
    int index_;
};

/// A model loaded onto a device and ready to be driven.
class InferenceSession {
public:
    virtual ~InferenceSession() = default;

    /// The package this session was built from — its declared inputs and outputs, training
    /// ranges, metrics and provenance travel with the running model.
    virtual const deploy::Package& get_package() const noexcept = 0;
    virtual Backend get_backend() const noexcept = 0;
    /// The device ordinal everything in this session runs on, resolved from the `Device` it was
    /// built with. Always a real index, never -1: `automatic()` resolves at construction.
    virtual int get_device() const noexcept = 0;

    /// Choose the grid the next inference fills.
    virtual void set_voxel_grid(std::array<std::uint32_t, 3> voxel_counts) = 0;

    /// Register a caller-owned input buffer holding values IN GRAPH UNITS.
    ///
    /// **The session does not normalize.** It cannot: a binding is registered once and re-read on
    /// every run, so a transform applied here would be silently skipped for every in-place edit
    /// afterwards — the one thing R-I1 promises works. Normalization belongs where the values are
    /// produced, which is why `FieldInference` converts the positions it generates
    /// (`deploy::apply`) and converts the outputs back (`deploy::invert`) itself. A caller binding
    /// a tensor directly converts it the same way, with the descriptor's own normalizer.
    ///
    /// The buffer arrives as a `MemoryRef`, so the same call binds a `std::vector`, an engine-owned
    /// D3D12 resource or a CUDA allocation and the caller never names a GPU type. A session refuses
    /// a domain it cannot read rather than copying behind the caller's back (R-I1); the reference
    /// must outlive the binding.
    virtual void bind_input(std::string_view name, std::shared_ptr<memory::MemoryRef> buffer) = 0;

    /// Register a caller-owned output buffer. The session writes into it and allocates nothing.
    virtual void bind_output(std::string_view name, std::shared_ptr<memory::MemoryRef> buffer) = 0;

    /// Run. Reports EVERY missing binding at once rather than the first.
    virtual void infer() = 0;

    /// A stage's parameters — the buffer every implementation of that stage reads.
    ///
    /// Materialised once when the session was built and valid for as long as it is. A stage with no
    /// parameters answers with an EMPTY buffer rather than throwing, so an implementation reads its
    /// weights the same way whether or not it has any; `not_found` means no such stage.
    /// `docs/custom-code.md` describes how each code form receives them.
    virtual const StageWeights& get_stage_weights(std::string_view stage) const = 0;
};

/// Build a session for a package on a backend and a device.
///
/// Selection order (R-F3): a specialised block matching the present device wins; otherwise the ONNX
/// graph runs. A stage with neither is refused by name (`Package::require_runnable_on`).
///
/// `device` decides which GPU everything uses. Pass `Device::of(*imported_buffer)` when a renderer
/// already owns the memory — that is what keeps the session and the buffer on the same card — or
/// `Device::ordinal(n)` to choose outright. Throws `FeatureDisabled` when the backend is compiled
/// out, `InvalidArgument` for an ordinal the backend does not have.
std::unique_ptr<InferenceSession> load(deploy::Package package, Backend backend,
                                       Device device = Device::automatic());

}  // namespace RadFiled3D::nn
