// What a compute backend can do, described without naming one.
//
// This exists so that rule 7 holds literally: **adding a backend adds files and edits none**. Before
// it, `memory/vk.cpp` included `backends/cuda.hpp` and switched on `Backend` to decide who could
// import a Vulkan allocation — so Vulkan's file knew about CUDA, and a new backend meant editing it.
// Now the graphics side asks the compute side, and neither names the other.
//
// NOTHING HERE NAMES A LIBRARY TYPE. No CUDA, no HIP, no ONNX Runtime — an implementation includes
// its own SDK inside its own translation unit, which is the only place those calls are allowed to
// be. `get_ort_allocator_name` is a plain string for exactly that reason: the ORT session needs to
// know which allocator a reference belongs to, and a string crosses that seam where an
// `Ort::MemoryInfo` would drag ONNX Runtime into every backend.
#pragma once

#include <RadFiled3D/nn/core/session.hpp>
#include <RadFiled3D/nn/core/stage.hpp>
#include <RadFiled3D/nn/memory/memory_ref.hpp>
#include <RadFiled3D/nn/memory/memory_ref.hpp>

#include <memory>
#include <string_view>

namespace RadFiled3D::nn {

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    virtual Backend get_backend() const noexcept = 0;
    /// The CMake option that compiles this backend in, for `FeatureDisabled` messages.
    virtual std::string_view get_feature() const noexcept = 0;
    /// Whether this build can run it at all.
    virtual bool available() const noexcept = 0;

    /// The domain this backend's own device allocations live in — `Domain::Cuda` for CUDA,
    /// `Domain::Hip` for ROCm, `Domain::Host` for the CPU provider.
    virtual memory::Domain get_memory_domain() const noexcept = 0;

    /// Whether memory from a graphics API can reach this backend AT ALL — a question about the two
    /// APIs, not about this build. Vulkan can reach CUDA and HIP; nothing reaches the CPU provider.
    /// `import_buffer` uses this to tell an impossible pairing (`UnsupportedInterop`) from one that
    /// is merely not compiled in (`FeatureDisabled`).
    virtual bool can_import_from(memory::Domain graphics) const noexcept = 0;

    /// Import an external graphics allocation into this backend's memory.
    ///
    /// The backend resolves the device itself from `buffer.device_uuid`, so a caller never has to
    /// map a Vulkan UUID onto a compute ordinal. The returned reference owns the mapping.
    virtual std::shared_ptr<memory::MemoryRef> import_external_memory(
        const memory::ExternalOrigin& buffer) const = 0;

    /// Make `memory` usable by THIS backend on `device`, whatever it currently is.
    ///
    /// The one call every bind path runs its input through, so no caller has to know what it is
    /// holding:
    ///
    ///   * already this backend's domain, on this device -> handed straight back, no allocation and
    ///     no copy. The common case, and it must stay free.
    ///   * imported from a graphics API that can be imported again -> imported here too, giving a
    ///     second view of THE SAME memory. Repeating the call returns the same reference rather than
    ///     mapping it twice.
    ///   * host memory -> handed back as it is. A runtime takes it with CPU memory info and does
    ///     the transfer itself; this is how a caller passes positions in, and always was.
    ///   * anything else -> throws, naming both domains and what would make it work.
    ///
    /// NOT virtual: it is expressed entirely in terms of `get_memory_domain` and
    /// `import_external_memory`, so a new backend gets it by existing.
    std::shared_ptr<memory::MemoryRef> adopt(std::shared_ptr<memory::MemoryRef> memory,
                                             int device) const;

    /// How many devices of this backend are present. 1 for the CPU provider; 0 when the backend is
    /// compiled in but no usable driver or card is there. What `Device::ordinal` is checked against.
    virtual int get_device_count() const noexcept = 0;

    /// The hardware family a package's specialised blocks are keyed by — `cuda`, `rocm`, or empty
    /// for a backend no compiled block targets.
    ///
    /// NOT the backend's own name. `Backend::TensorRt` and `Backend::Cuda` are two providers on ONE
    /// piece of hardware, so a `cuda_ptx` block runs under both; keying selection on `to_string()`
    /// would refuse a CUDA kernel on the TensorRT provider while claiming the hardware could not run
    /// it. This is the same vocabulary `BlockKind::get_target()` answers in, which is what makes the
    /// comparison meaningful.
    virtual std::string_view get_block_target() const noexcept = 0;

    /// The architecture of a device of this backend — `sm_86` for CUDA, `gfx1100` for ROCm — or
    /// empty when the backend has no such notion or cannot be reached.
    ///
    /// This is what decides whether a package's specialised blocks fit the machine in front of it,
    /// so it is a question one backend asks another rather than a switch over `Backend` somewhere
    /// else. A build with the backend compiled out answers empty rather than throwing: not knowing
    /// the architecture is an ordinary state, and the portable ONNX form does not care.
    virtual std::string get_device_arch(int device) const noexcept = 0;

    // ── memory this backend owns ────────────────────────────────────────────────────────────────
    //
    // The intermediates between stages and a stage's weights live HERE, in the backend's own domain,
    // so a kernel can read them directly and ONNX Runtime can bind them without a host round trip.
    // Allocated once when the grid is chosen and reused by every inference.

    /// Allocate `bytes` in this backend's memory domain. Zero bytes is legal and yields a valid,
    /// empty reference — a stage with no weights still HAS the buffer.
    virtual std::shared_ptr<memory::MemoryRef> allocate(std::uint64_t bytes, int device) const = 0;

    /// Copy within this backend's domain. Both references must be in `get_memory_domain()`.
    virtual void copy_within(memory::MemoryRef& dst, std::uint64_t dst_offset,
                             const memory::MemoryRef& src, std::uint64_t src_offset,
                             std::uint64_t bytes) const = 0;

    /// Fill `bytes` of a reference in this domain with host data.
    virtual void upload(memory::MemoryRef& dst, const void* source, std::uint64_t bytes) const = 0;

    // ── compiled stages ─────────────────────────────────────────────────────────────────────────

    /// Whether this backend can run a stage that is a compiled kernel rather than a graph.
    virtual bool can_launch_kernels() const noexcept = 0;

    /// Build a stage from a block of machine code.
    ///
    /// `kind` is the block kind's wire name (`cuda_ptx`, `cuda_cubin`, …) — a string, not a type, for
    /// the same reason the ORT allocator is one. The `launch` descriptor says which symbol to call
    /// and in what shape. The whole `composition` is passed, not just the stage, because a blob of
    /// machine code declares no tensors and no shapes: what its arguments ARE comes from the stage's
    /// ports, and how big they are from the buffer table. Throws `FeatureDisabled` when this backend
    /// cannot launch kernels at all.
    virtual std::unique_ptr<Stage> make_kernel_stage(const deploy::Composition& composition,
                                                     const deploy::Stage& stage, std::string_view kind,
                                                     deploy::byte_view code,
                                                     const deploy::KernelLaunch& launch,
                                                     int device) const = 0;

    /// The name ONNX Runtime knows this backend's allocator by ("Cuda", "Hip", ...), or empty for
    /// one whose memory is ordinary host memory. A plain string so that no backend has to include
    /// ONNX Runtime to say it.
    virtual std::string_view get_ort_allocator_name() const noexcept = 0;
};

/// The backend behind a `Backend` value.
///
/// Lives with the rest of the backend vocabulary (`core/session.hpp`), which is where CLAUDE.md
/// already says that vocabulary is defined once — the registry is part of it. This is the single
/// place that enumerates the backends, and it enumerates nothing else.
const ComputeBackend& get_compute_backend(Backend backend);

}  // namespace RadFiled3D::nn
