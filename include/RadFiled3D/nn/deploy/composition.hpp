// How a package's stages compose, and what each one needs to run.
//
// A real model is not one graph. `PBRFNet` factors into three:
//
//     encoding_config(region_width)                 -> region_state
//     beam_encoder(direction, distance, spectrum)   -> latent
//     trunk(position, latent, region_state)         -> flux, spectrum
//
// and the container used to record NOWHERE how they connect. A consumer had to know, out of band,
// that the encoder's output feeds the trunk — and for the packages that existed it could not even
// guess, because the exporter named that output `linear_5` while the trunk calls its input `latent`.
//
// THREE things are recorded, and only three:
//
//   * **Buffers** — the named memory between stages. A stage writes into one and any LATER stage
//     reads from it, so a value computed once reaches every stage that needs it without passing
//     through the caller. They are allocated when the grid is chosen and live as long as it does.
//   * **Stages** — which part runs, in what order, HOW OFTEN, and which of its tensors map to which
//     buffers. `Invocation` is genuinely new information rather than a derivable fact: `beam_encoder`
//     declares a batch axis on every input (`direction` is `[batch, 3]`) and still runs exactly once
//     per field, because a beam is constant over the volume. Nothing in the graph says that.
//   * **Launch descriptors** — for a stage implemented by a compiled kernel rather than a graph,
//     how to call it. A graph needs none; the runtime asks ONNX Runtime.
//
// WHAT IS NOT RECORDED HERE: which machine code a stage has. That is already in the blocks. A stage
// named `hashgrid` is implemented by whatever blocks share its name — `onnx:hashgrid`,
// `cuda_ptx:hashgrid`, `hsaco:hashgrid`, `spirv:hashgrid` — and `Package::select_block` picks the
// one that runs on the present device, preferring an exact architecture match over a portable form.
// Listing the variants here as well would be a second source of truth about what the package holds.
//
// It travels as a BLOCK (`BlockKind::Composition`), not as a metadata field. Rule 5 guarantees that
// a block a build does not recognise is skipped by its length and written back verbatim; the
// metadata region has no such guarantee, so a rewrite by a composition-unaware tool would silently
// drop the wiring and leave a package whose stages can no longer be connected.
#pragma once

#include <RadFiled3D/nn/deploy/codec.hpp>
#include <RadFiled3D/nn/deploy/descriptor.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace RadFiled3D::nn::deploy {

struct BlockKind;

/// How often a stage runs, relative to one field inference.
enum class Invocation : std::uint8_t {
    /// Once per field: the beam encoder for a beam, the encoding config for a grid. What it writes
    /// is computed once and reused for every query.
    Once = 0,
    /// Once per query batch — the trunk, evaluated at every voxel.
    PerQuery = 1,
};

std::string_view to_string(Invocation invocation) noexcept;
/// Throws `InvalidArgument` naming the accepted vocabulary.
Invocation invocation_from_name(std::string_view name);

/// A named block of memory between stages.
///
/// `elements` is per ROW; how many rows it holds is derived when the stages are wired, from the
/// shapes the producing and consuming stages actually declare — a value written once and read per
/// query is repeated across the batch, one written and read at the same extent is carried as-is.
/// Storing a row count here would be a second copy of something the shapes already say.
struct Buffer {
    std::string name;
    std::uint64_t elements = 0;
    DType dtype = DType::F32;

    bool operator==(const Buffer&) const = default;

    void encode(Writer& w) const;
    static Buffer decode(Reader& r);
};

/// Where a stage's parameters live.
///
/// ONE buffer, belonging to ONE stage, and shared by that stage's IMPLEMENTATIONS. A stage may be
/// evaluated by a PTX kernel on NVIDIA, an HSACO object on AMD and an ONNX graph anywhere; those are
/// three ways to compute the same function, so they read one set of numbers. A copy per
/// implementation would be three things that can drift apart, and a package whose CUDA path and CPU
/// path quietly disagree is the worst failure this format could allow.
///
/// It lives in a `weights` block sharing the stage's name — which is why nothing here records a
/// block name. The stage owns its parameters; no other stage reads them.
///
/// Every stage HAS this buffer and it MAY BE EMPTY. A stage with no parameters — an activation, a
/// reduction, a gather — has a zero-length one, and `InferenceSession::get_stage_weights` hands back
/// an empty view rather than nothing, so an implementation reads its weights the same way whether or
/// not it has any. Only the storage is elided: no block is written for an empty buffer, because a
/// zero-length block and an absent one say the same thing and one of them costs bytes.
struct Weights {
    /// How many values the block holds, so a consumer knows the layout without parsing the payload.
    std::uint64_t elements = 0;
    /// The element type every implementation of the stage agrees on — `f32`, `f16` for a
    /// half-precision kernel, `i32`, `u8`. ONE type for the whole buffer, because it is one buffer:
    /// a PTX kernel reading it as `f16` while the ONNX graph reads `f32` is the drift this design
    /// exists to prevent.
    DType dtype = DType::F32;
    /// The external-data file name an ONNX implementation refers to these by.
    ///
    /// An ONNX graph exported with external data does not embed its initializers; it records the
    /// name of a file to find them in. Recording that name here is what lets the runtime satisfy it
    /// FROM THE PACKAGE (`AddExternalInitializersFromFilesInMemory`) instead of requiring a loose
    /// file beside the `.rf3m`, which would defeat the point of a self-contained container. Empty
    /// when the graph embeds its own initializers, which is the ordinary case.
    std::string onnx_external_file;

    bool operator==(const Weights&) const = default;

    void encode(Writer& w) const;
    static Weights decode(Reader& r);
};

/// One of a stage's tensors, and the buffer it is attached to.
struct Port {
    /// The tensor name the implementation uses — a graph input/output, or a kernel parameter.
    std::string tensor;
    /// The `Buffer` it reads from or writes to.
    std::string buffer;

    bool operator==(const Port&) const = default;

    void encode(Writer& w) const;
    static Port decode(Reader& r);
};

/// Which compiled form a launch descriptor belongs to: `cuda_ptx` on `sm_90`, `hsaco` anywhere, …
struct BlockRef {
    /// The wire name of the block kind (`cuda_ptx`, `hsaco`, `spirv`, …).
    std::string kind;
    /// The architecture that variant was built for. Empty means "any this kind runs on".
    std::string target_arch;

    bool operator==(const BlockRef&) const = default;

    void encode(Writer& w) const;
    static BlockRef decode(Reader& r);
};

/// How to call one compiled implementation of a stage.
///
/// A graph carries its own calling convention and needs none of this; a kernel blob is just bytes,
/// and without an entry point and a launch shape a consumer has code it cannot run. Kept per
/// implementation rather than per stage because the variants genuinely differ — a CUDA kernel and an
/// HSACO one need not share an entry symbol or a block size.
///
/// The ARGUMENT order a launched kernel receives — weights first, then inputs, then outputs, then
/// the row count — is the contract in `docs/custom-code.md` §4. It is written down in exactly one
/// place because a blob of machine code has no way to ask.
struct KernelLaunch {
    /// The symbol to launch in the loaded module.
    std::string entry_point;
    /// Threads per block / work-group size.
    std::array<std::uint32_t, 3> block_size{256, 1, 1};
    /// Blocks per grid. A zero component is DERIVED from the row count the stage runs at —
    /// `ceil(rows / block_size)` on x, 1 elsewhere — so a kernel that is one thread per query needs
    /// no grid recorded and still adapts to the caller's chosen resolution.
    std::array<std::uint32_t, 3> grid_size{0, 0, 0};
    std::uint32_t shared_memory_bytes = 0;

    bool operator==(const KernelLaunch&) const = default;

    /// The grid this launch uses for `rows` items, with the derived components filled in.
    std::array<std::uint32_t, 3> get_grid_for(std::uint64_t rows) const noexcept;

    void encode(Writer& w) const;
    static KernelLaunch decode(Reader& r);
};

/// One part of the model.
struct Stage {
    /// Identifies the stage AND names the blocks that implement it.
    std::string name;
    Invocation invocation = Invocation::PerQuery;
    /// Tensors fed from a buffer an earlier stage wrote. A tensor with no port here is the
    /// CALLER's to bind.
    std::vector<Port> reads;
    /// Tensors written into a buffer for later stages. A tensor with no port here is a RESULT, and
    /// the caller binds somewhere to receive it.
    std::vector<Port> writes;
    /// How to launch each compiled implementation. Empty for a stage that only ever runs as a graph.
    std::vector<std::pair<BlockRef, KernelLaunch>> launches;
    /// The parameters every implementation of this stage reads. Empty for a stage that has none.
    Weights weights;

    bool operator==(const Stage&) const = default;

    /// The launch descriptor for a compiled form, or null when this stage has none for it.
    const KernelLaunch* get_launch_for(std::string_view kind, std::string_view arch) const noexcept;

    void encode(Writer& w) const;
    static Stage decode(Reader& r);
};

/// The whole wiring: the memory between stages, and the stages in execution order.
struct Composition {
    std::vector<Buffer> buffers;
    /// Execution order, first to last. Recorded rather than derived by a topological sort, so a
    /// package says outright what a consumer should do instead of asking it to infer one of several
    /// valid orders.
    std::vector<Stage> stages;

    bool operator==(const Composition&) const = default;

    const Buffer* get_buffer(std::string_view name) const noexcept;
    const Stage* get_stage(std::string_view name) const noexcept;
    /// The stage that writes `buffer`, or null.
    const Stage* get_writer_of(std::string_view buffer) const noexcept;
    /// Position of `stage` in the execution order, or -1.
    int get_order_of(std::string_view stage) const noexcept;

    /// Structural checks, run on write and on read. Throws `InvalidPackage`.
    ///
    /// `block_names` is every executable block the package carries and `weights_blocks` every
    /// `weights` block, so a stage nothing implements — or one whose parameters are not there — is
    /// caught. Tensor names cannot be checked here — that would mean parsing ONNX, which
    /// `rfnn::deploy` deliberately cannot do (rule 3: reading a package must not require a runtime) —
    /// so a wrong tensor name surfaces when a session binds it, by name.
    void validate(const std::vector<std::string_view>& block_names,
                  const std::vector<std::string_view>& weights_blocks) const;

    void encode(Writer& w) const;
    static Composition decode(Reader& r);

    bytes to_bytes() const;
    static Composition from_bytes(byte_view bytes);
};

/// The block name the composition travels under. One per package.
inline constexpr std::string_view kCompositionBlock = "composition";

}  // namespace RadFiled3D::nn::deploy
