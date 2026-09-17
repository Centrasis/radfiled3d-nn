// The RF3M container.
//
// Layout (little-endian throughout, requirements.md R-F1):
//
//   [4]   magic            "RF3M"
//   [u32] version          the file version; 1 = this layout (deploy::v1, kVersion)
//   [32]  digest           BLAKE3 of every byte after this field
//   [u64] metadata_bytes   skip this many to reach the first block
//   [metadata_bytes]       provenance, I/O descriptors, geometry, metrics, RadFiled3D provenance
//   [u32] block_count
//   block_count x:
//     [u64] block_bytes    everything after this field, so an unwanted block is one seek
//     [str] kind           "onnx", "cuda_ptx", "cuda_cubin", "opencl", ...
//     [str] name           "trunk", "beam_encoder", ...
//     [str] target_arch    "sm_90"; empty = runs anywhere its kind runs
//     [rest of block]      the payload
//
// `[str]` is `[u32 len][utf-8 bytes]`, not NUL-terminated.
//
// Four properties follow, and each fixes a defect of the original format (requirements.md §2):
//
//   * Skip what you did not ask for. `metadata_bytes` jumps straight past the header, and every
//     block declares its own length, so listing what a 400 MB package contains reads a few hundred
//     bytes: magic, version, one seek, then `[len][kind][name]` per block (D4).
//   * One concept for every executable form. An ONNX graph and a hand-tuned CUDA kernel are the
//     same thing at this level — a named, typed block — so a package can carry both and the loader
//     picks by kind. Adding a kind costs nothing: an unrecognised one is skipped and preserved (D5).
//   * Forward compatibility. Unknown block kinds survive a rewrite, and each half of the metadata
//     is length-prefixed, so a producer with extra fields does not desynchronise a reader.
//   * Integrity. The digest tells a truncated or edited package from a good one before anything
//     tries to parse it (D6).
#pragma once

#include <RadFiled3D/nn/deploy/codec.hpp>
#include <RadFiled3D/nn/deploy/composition.hpp>
#include <RadFiled3D/nn/deploy/descriptor.hpp>
#include <RadFiled3D/nn/core/types.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace RadFiled3D::nn::deploy {

// Declared in deploy/version.hpp, which includes this header — hence the forward declaration
// rather than an include.
enum class FormatVersion : std::uint32_t;

inline constexpr std::array<std::uint8_t, 4> kMagic = {'R', 'F', '3', 'M'};
/// File version written by this library.
inline constexpr std::uint32_t kVersion = 1;
inline constexpr std::size_t kDigestBytes = 32;
/// magic + version + digest + metadata length word.
inline constexpr std::size_t kHeaderBytes = 4 + 4 + kDigestBytes + 8;

/// The block every package must carry: the ONNX graph that consumes the beam parameters.
inline constexpr std::string_view kTrunkGraph = "trunk";
/// Conventional name for a beam-vector-to-latent encoder in a two-graph model.
inline constexpr std::string_view kBeamEncoderGraph = "beam_encoder";

/// The executable form a block holds.
///
/// Names on the wire, types in the API: a kind this build has never heard of decodes to `Custom`,
/// survives a rewrite verbatim, and is simply never selected — the same rule `Semantic` follows.
/// A package can therefore gain an accelerator without a format version.
struct BlockKind {
    enum Kind : std::uint8_t {
        /// An ONNX graph. The portable form, and the fallback every package must have.
        Onnx,
        /// NVIDIA PTX, JIT-compiled by the driver — runs on any architecture of its target.
        CudaPtx,
        /// NVIDIA cubin, compiled for one architecture.
        CudaCubin,
        /// AMD GPU code object.
        Hsaco,
        /// OpenCL C source.
        OpenCl,
        /// SPIR-V — the portable IR both Level Zero and OpenCL consume, and how an Intel GPU kernel
        /// travels. A package may CARRY one today and no backend selects it, because this library
        /// has no Intel compute backend; `Backend` is the list of what can actually run.
        SpirV,
        /// NOT an executable form: a stage's PARAMETERS, shared by every implementation of it.
        ///
        /// A block rather than a metadata field for the same reason a graph is one: it is a large,
        /// opaque, named payload a scan must be able to skip by its length. Its `name` is what a
        /// stage's `Weights::block` refers to.
        Weights,
        /// NOT an executable form: how the package's graphs compose (`composition.hpp`).
        ///
        /// It is a block rather than a metadata field precisely so rule 5 covers it — a tool that
        /// does not understand the wiring still writes it back verbatim instead of leaving a package
        /// whose graphs can no longer be connected.
        Composition,
        Custom,
    };

    Kind kind = Custom;
    std::string custom_name;

    BlockKind() = default;
    BlockKind(Kind k) : kind(k) {}  // NOLINT(google-explicit-constructor)

    static BlockKind from_name(std::string_view name);
    std::string_view get_name() const noexcept;

    /// Whether this build recognises the kind at all.
    bool is_known() const noexcept { return kind != Custom; }

    /// Whether the block is a form of the model that can be RUN. `Composition` is not — it describes
    /// how the executable blocks fit together — and neither is `Weights`, which is what they read.
    /// `validate()` leans on this so that adding a non-executable kind never accidentally satisfies
    /// "a package must carry a runnable graph".
    bool is_executable() const noexcept {
        return kind != Composition && kind != Weights && kind != Custom;
    }

    /// The accelerator family the kind needs, or empty for the portable ONNX form.
    std::string_view get_target() const noexcept;

    bool operator==(const BlockKind&) const = default;
};

/// One executable form of the model: a length-prefixed, typed, named payload.
struct Block {
    BlockKind kind;
    /// Which part of the model this is — `trunk`, `beam_encoder`, ...
    std::string name;
    /// Architecture it was built for ("sm_90"). Empty means it runs anywhere its kind runs.
    std::string target_arch;
    bytes payload;

    static Block onnx(std::string name, bytes graph);

    /// Whether this block can run on a device of the given target and architecture. An
    /// architecture-agnostic block (empty `target_arch`, e.g. PTX the driver JITs) matches any
    /// architecture of its target.
    bool runs_on(std::string_view target, std::string_view arch) const noexcept;

    bool operator==(const Block&) const = default;
};

/// What a block declares, without its payload — the result of a quick scan.
struct BlockInfo {
    BlockKind kind;
    std::string name;
    std::string target_arch;
    std::uint64_t payload_bytes = 0;

    bool operator==(const BlockInfo&) const = default;
};

/// What the model was trained on. No per-simulation tube metadata: that belongs to a field, not to
/// a model that generalises over many.
struct Provenance {
    std::string dataset;
    std::string software;
    std::string physics;
    /// ISO-8601 UTC, or empty.
    std::string created;

    bool operator==(const Provenance&) const = default;

    void encode(Writer& w) const;
    static Provenance decode(Reader& r);
};

struct Voxelization {
    std::array<std::uint32_t, 3> voxel_counts{};
    std::array<float, 3> voxel_dimensions_m{};

    bool operator==(const Voxelization&) const = default;
};

/// The metric box a normalised position maps into, and — only where the architecture genuinely
/// fixes one — the voxel grid the model emits.
///
/// A voxelization is normally ABSENT: the inference grid is chosen by the caller from available
/// GPU memory and varies across a dataset, so recording one would make a point-field model silently
/// wrong at every other grid (R-F4). It is present only for a whole-volume model.
struct Geometry {
    /// Edge lengths in metres of the box that normalised positions span. Zero means unknown.
    std::array<float, 3> field_dimensions_m{};
    std::optional<Voxelization> voxelization;

    bool operator==(const Geometry&) const = default;

    void encode(Writer& w) const;
    static Geometry decode(Reader& r);
};

// ── RadFiled3D provenance ─────────────────────────────────────────────────────────────────────────
//
// The RadFiled3D metadata of the data a model was trained on, carried inside the package so a
// `.rf3` written from an inference result can say where it came from in the same vocabulary a
// simulated field uses (R-F8).
//
// The fields are encoded one at a time, NEVER as a struct image. RadFiled3D holds this as a
// `#pragma pack(4)` POD, and copying that POD into the container would put a host-endian,
// ABI-dependent memory image inside a format that is little-endian and length-prefixed everywhere
// else — a file whose meaning depended on the compiler that wrote it. The conversion to and from
// RadFiled3D's own type lives in `RadFiled3D/nn/core/rf3_metadata.hpp`; this header knows nothing of it.

struct Rf3XRayTube {
    std::array<float, 3> radiation_direction{};
    std::array<float, 3> radiation_origin{};
    float max_energy_ev = 0.f;
    std::string tube_id;

    bool operator==(const Rf3XRayTube&) const = default;
};

/// The Monte-Carlo run a training field came from.
struct Rf3Simulation {
    std::uint64_t primary_particle_count = 0;
    std::string geometry;
    std::string physics_list;
    Rf3XRayTube tube;

    bool operator==(const Rf3Simulation&) const = default;
};

/// The software that wrote the training data.
struct Rf3Software {
    std::string name;
    std::string version;
    std::string repository;
    std::string commit;
    std::string doi;

    bool operator==(const Rf3Software&) const = default;
};

struct Rf3Metadata {
    /// RadFiled3D `StoreVersion` this describes; 1 = V1.
    std::uint32_t store_version = 1;
    Rf3Simulation simulation;
    Rf3Software software;
    /// RadFiled3D's dynamic metadata layers, as its own channel serialiser wrote them.
    ///
    /// Genuinely opaque — arbitrary named voxel layers that only RadFiled3D can interpret — so it
    /// travels as bytes. This is the TAIL only: the fixed header RadFiled3D prepends when it
    /// serialises is not repeated here, because it is already carried, typed, in the fields above.
    /// Storing both would be two sources of truth with no rule for which wins.
    bytes dynamic;

    bool operator==(const Rf3Metadata&) const = default;

    void encode(Writer& w) const;
    static Rf3Metadata decode(Reader& r);
};

/// Which of the two shapes of model a package carries — the distinction that decides how it is
/// driven, and the one the training framework draws between its FCNN and CNN architectures.
///
/// It is a QUESTION ASKED OF THE DESCRIPTORS, not a field: a model declaring a `Position` input is
/// queried a point at a time, and one that does not emits the grid in a single run. Nothing in the
/// container records the kind, so nothing can disagree with the interface it declares.
enum class ModelKind : std::uint8_t {
    /// Queried per point: the runtime supplies one position per voxel and the model answers for it.
    /// The grid is the CALLER's choice, which is why a package like this may record no voxelization.
    /// PBRFNet and TPBRFNet are this shape — fully-connected networks over a coordinate.
    VoxelWise,
    /// Emits the whole field in one run, so the grid is the MODEL's, fixed by its architecture and
    /// recorded as `Geometry::voxelization`. A convolutional network is this shape.
    WholeVolume,
};

std::string_view to_string(ModelKind kind) noexcept;

// ── the package ───────────────────────────────────────────────────────────────────────────────────

/// What a package declares, read without touching a payload.
///
/// This is the quick scan: magic, version, one seek past the metadata, then `[len][kind][name]` per
/// block. Listing a directory of models — their training ranges, metrics, and which accelerators
/// they carry code for — costs a few kilobytes per file whatever the payloads weigh.
struct Metadata {
    Provenance provenance;
    std::vector<TensorDescriptor> io;
    Geometry geometry;
    std::map<std::string, double> metrics;
    std::optional<Rf3Metadata> rf3_metadata;
    std::vector<BlockInfo> blocks;
    /// How the graphs compose, when the package says.
    ///
    /// The ONE payload a scan reads. It is a few hundred bytes describing the model rather than
    /// being part of it, and a listing that showed a package's graphs without showing how they
    /// connect would answer the less useful half of the question. Every other payload is skipped by
    /// its length, which is what keeps the scan cheap whatever the model weighs.
    std::optional<Composition> composition;

    bool operator==(const Metadata&) const = default;
};

/// One decoded RF3M package.
struct Package {
    Provenance provenance;
    /// Inputs and outputs, in declaration order.
    std::vector<TensorDescriptor> io;
    Geometry geometry;
    /// Test metrics recorded after training.
    std::map<std::string, double> metrics;
    /// The RadFiled3D provenance of the training data, if the producer recorded it.
    std::optional<Rf3Metadata> rf3_metadata;
    /// The executable forms of the model, in file order. At least one ONNX block named `trunk`.
    std::vector<Block> blocks;

    bool operator==(const Package&) const = default;

    // ── queries ─────────────────────────────────────────────────────────────────────────────────

    std::vector<const TensorDescriptor*> get_inputs() const;
    std::vector<const TensorDescriptor*> get_outputs() const;
    const TensorDescriptor* get_tensor(std::string_view name) const noexcept;

    /// The first tensor carrying a semantic, if any. Consumers ask by meaning rather than by name
    /// when they do not care what the producer called it.
    const TensorDescriptor* get_tensor_with(Role role, const Semantic& semantic) const noexcept;

    /// A model queried at explicit positions is per-voxel; one that is not emits a whole volume in
    /// one shot. This single fact decides which runtime drives it — the rule is inherited from the
    /// original, but it now reads off a descriptor instead of a flag.
    bool is_voxelwise() const noexcept;

    /// The same distinction as a name, for reporting and for dispatch.
    ///
    /// DERIVED, never stored. A `ModelKind` written into the container could contradict the presence
    /// of a `Position` input, and two sources of truth with no rule for which wins is the defect
    /// this format exists to close (D3). It is computed from `is_voxelwise()` and nothing else.
    ModelKind get_model_kind() const noexcept;

    /// Semantics this build does not know. Not an error — a consumer reports them rather than
    /// refusing a package it can still describe.
    std::vector<std::string_view> get_unknown_semantics() const;

    /// The ONNX graph called `name`, if the package carries one.
    std::optional<byte_view> get_graph(std::string_view name) const noexcept;

    /// A stage's parameters, shared by every implementation of it.
    ///
    /// ALWAYS answers: a stage with no parameters gets an empty view, not nothing, so an
    /// implementation reads its weights the same way whether or not it has any. This is the one
    /// entry point — `rfnn::deploy` stores them, `InferenceSession::get_stage_weights` delivers
    /// them, and `docs/custom-code.md` describes how each code form receives them.
    byte_view get_weights(std::string_view stage) const noexcept;
    /// Names of the `weights` blocks the package carries.
    std::vector<std::string_view> get_weights_block_names() const;

    /// Whether the model runs anywhere: every stage has a portable ONNX form.
    ///
    /// A package may deliberately be non-portable — a stage implemented only as `cuda_ptx` is a
    /// legitimate thing to ship, and is why `require_runnable_on` exists. This answers the question
    /// a producer wants answered before it ships one, rather than refusing to write it.
    bool is_portable() const noexcept;

    /// Refuse a package the present device cannot run, naming EVERY stage responsible.
    ///
    /// `target` is the hardware FAMILY specialised blocks are keyed by (`cuda`, `rocm`, or empty for
    /// one nothing targets) and `arch` the device's architecture (`sm_86`). A stage is runnable when
    /// the package holds an ONNX graph of its name — the portable path, open to every backend — or a
    /// specialised block that `select_block` picks for this device. Anything else means the hardware
    /// is missing code that nothing can substitute.
    ///
    /// `label` is what the message calls the thing that cannot run it, and exists because the family
    /// is often not what a user would recognise: the CPU provider targets no family at all, and
    /// "none of which runs on /unknown architecture" names nothing. The caller passes the backend's
    /// own name. Defaults to `target`.
    ///
    /// Reports every unsupported stage at once, each with the forms it does carry, because a user
    /// told about one missing kernel at a time learns the answer one rebuild at a time.
    void require_runnable_on(std::string_view target, std::string_view arch,
                             std::string_view label = {}) const;

    /// How this package's graphs compose, when it says.
    ///
    /// Absent means the package is a single `trunk` driven directly — which is what every package
    /// written before this existed is, and what a single-graph model still is. A consumer that finds
    /// a composition must honour it; one that finds none runs the trunk.
    std::optional<Composition> get_composition() const;
    /// Names of the ONNX blocks — the portable forms.
    std::vector<std::string_view> get_graph_names() const;
    /// Names of every block that can be RUN, in any form, deduplicated. This is what a composition
    /// stage is checked against: a stage may be implemented by a kernel rather than a graph.
    std::vector<std::string_view> get_executable_block_names() const;
    /// Every executable form the package carries under `name`, as `kind(arch)` strings, for the
    /// error message when none of them fits the present device.
    std::vector<std::string> describe_blocks(std::string_view name) const;

    /// The block to execute for `name` on a device, or null to fall back to the ONNX graph.
    ///
    /// An exact architecture match beats an architecture-agnostic one: it was compiled for this
    /// device, where the agnostic blob still has to be JIT-compiled.
    const Block* select_block(std::string_view name, std::string_view target,
                              std::string_view arch) const noexcept;

    /// Structural checks that must hold for a package to be loadable. Run on write as well as on
    /// read, so a package that could not load is never produced in the first place.
    void validate() const;

    // ── reading ─────────────────────────────────────────────────────────────────────────────────

    static Package read(byte_view bytes);
    static Package read_file(const std::filesystem::path& path);

    /// Read the header and scan the blocks, touching no payload.
    static Metadata read_metadata(byte_view bytes);
    static Metadata read_metadata_file(const std::filesystem::path& path);

    // ── writing ─────────────────────────────────────────────────────────────────────────────────

    /// Serialise in the version this library writes (`kWriteVersion`).
    bytes to_bytes() const;
    /// Serialise in a named version. A version this build cannot write throws `UnsupportedVersion`
    /// rather than quietly emitting a different one.
    bytes to_bytes(FormatVersion version) const;
    void write_file(const std::filesystem::path& path) const;
};

/// Read a whole file into memory. Shared by the package readers and by nothing else.
bytes read_all(const std::filesystem::path& path);

}  // namespace RadFiled3D::nn::deploy
