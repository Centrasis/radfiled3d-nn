// Authoring a package.
//
// This is the API the training framework calls to export a model (requirements.md R-P1). It exists
// so that a producer never assembles a `Package` field by field and never learns the byte layout:
// it describes the model, hands over the graphs, and the builder refuses anything that would not
// load.
//
// Adding an output the library has never heard of is an ordinary call:
//
//     PackageBuilder b;
//     b.provenance("xray-scatter-v3", "radfield3d-nn 2.1", "G4EmStandardPhysics_option4")
//      .field_dimensions_m({1.f, 1.f, 1.f})
//      .input("position", Semantic::Position, {3}).unit("m").done()
//      .input("beam_direction", Semantic::BeamDirection, {2}).unit("rad")
//          .range(MinMax{-3.14159, 3.14159}).normalizer(LinearSym{-3.14159, 3.14159}).done()
//      .output("flux", Semantic::Flux, {1}).normalizer(LogScale{1e-12, 30.0}).done()
//      // A quantity this library does not know needs no change to this library.
//      .output("direction_distribution", Semantic::from_name("direction_distribution"), {16, 32}).done()
//      .graph("trunk", onnx_bytes)
//      .metric("air_kerma_smape", 0.043);
//     Package package = b.build();
#pragma once

#include <RadFiled3D/nn/deploy/package.hpp>

namespace RadFiled3D::nn::deploy {

class PackageBuilder;

/// A stage under construction. `done()` returns to the package builder.
///
/// Every call appends to the composition block as it happens, so the builder's invariant — at most
/// one composition block, always consistent — holds at every point rather than only at `build()`.
class StageBuilder {
public:
    /// Feed one of this stage's tensors from a named buffer an EARLIER stage wrote.
    StageBuilder& reads(std::string tensor, std::string buffer);
    /// Write one of this stage's tensors into a named buffer for later stages. A tensor with no
    /// `writes` is a result the caller binds.
    StageBuilder& writes(std::string tensor, std::string buffer);
    /// How to launch one compiled form of this stage. Needed only for a stage implemented by a
    /// kernel block; a graph carries its own calling convention and must not declare one.
    StageBuilder& launch(BlockKind kind, std::string target_arch, KernelLaunch launch);

    /// Attach this stage's parameters — the one buffer EVERY implementation of this stage reads.
    ///
    /// Stored once, under a `weights` block sharing the stage's name, and handed to the PTX kernel,
    /// the HSACO object, the SPIR-V kernel and the ONNX graph alike. `dtype` is the element type they
    /// agree on — `f16` for a half-precision kernel, `f32` otherwise. `onnx_external_file` is the
    /// external-data file name an ONNX implementation refers to them by, and is empty for a graph
    /// that embeds its own initializers.
    StageBuilder& weights(bytes payload, std::uint64_t elements, DType dtype = DType::F32,
                          std::string onnx_external_file = {});
    PackageBuilder& done();

private:
    friend class PackageBuilder;
    StageBuilder(PackageBuilder& builder, std::string name) : builder_(builder), name_(std::move(name)) {}
    /// Read the stage back out of the block, apply `edit`, write it back.
    template <typename F>
    StageBuilder& amend(F&& edit);
    PackageBuilder& builder_;
    std::string name_;
};

/// A descriptor under construction. `done()` returns to the package builder.
class TensorBuilder {
public:
    TensorBuilder& unit(std::string unit);
    TensorBuilder& range(Range range);
    TensorBuilder& normalizer(Normalizer normalizer);
    TensorBuilder& dtype(DType dtype);
    TensorBuilder& optional(bool optional = true);
    PackageBuilder& done();

private:
    friend class PackageBuilder;
    TensorBuilder(PackageBuilder& builder, TensorDescriptor descriptor)
        : builder_(builder), descriptor_(std::move(descriptor)) {}
    PackageBuilder& builder_;
    TensorDescriptor descriptor_;
};

class PackageBuilder {
public:
    PackageBuilder() = default;
    explicit PackageBuilder(Geometry geometry) { package_.geometry = std::move(geometry); }

    PackageBuilder& provenance(std::string dataset, std::string software, std::string physics);
    /// ISO-8601 UTC. Supplied by the caller rather than read from the clock, so that building the
    /// same model twice can produce the same bytes.
    PackageBuilder& created(std::string timestamp);
    /// The metric box that normalised positions span.
    PackageBuilder& field_dimensions_m(std::array<float, 3> dims);
    /// Record a fixed voxel grid. Only valid for a whole-volume model; `build()` rejects it on a
    /// model queried per position (R-F4).
    PackageBuilder& voxelization(std::array<std::uint32_t, 3> voxel_counts,
                                 std::array<float, 3> voxel_dimensions_m);

    TensorBuilder input(std::string name, Semantic semantic, std::vector<std::uint32_t> shape);
    TensorBuilder output(std::string name, Semantic semantic, std::vector<std::uint32_t> shape);
    PackageBuilder& descriptor(TensorDescriptor descriptor);

    /// Attach an ONNX graph. Every package needs one called `trunk`.
    PackageBuilder& graph(std::string name, bytes onnx);
    /// Attach a specialised executable form of the same model — PTX, a cubin, OpenCL — beside the
    /// portable graph. `target_arch` may be empty for a form that runs on any architecture of its
    /// kind.
    PackageBuilder& block(BlockKind kind, std::string name, std::string target_arch, bytes payload);
    /// Embed the RadFiled3D metadata of the training data, so a `.rf3` written from an inference
    /// result can carry the tube, geometry and physics list the model was trained on.
    PackageBuilder& rf3_metadata(Rf3Metadata metadata);

    /// Declare a named buffer between stages: `elements` per row, and the element type.
    ///
    /// How many ROWS it holds is not recorded — it is derived when the stages are wired, from the
    /// shapes the producer and consumer declare, so a value written once and read at every query is
    /// repeated and one read at the same extent is carried as-is.
    PackageBuilder& buffer(std::string name, std::uint64_t elements, DType dtype = DType::F32);

    /// Record that a stage runs, and how often. Call once per stage, IN EXECUTION ORDER; the
    /// `trunk` runs last and per query.
    ///
    /// The name is the stage's identity AND the name of the blocks that implement it — an
    /// `onnx:hashgrid` graph, a `cuda_ptx:hashgrid` kernel, or both. Which one runs is decided
    /// against the device at load, so a stage carrying only a kernel is a package that loads on
    /// that hardware and refuses, by name, on any other.
    ///
    /// A package with no stages at all composes the way every package always has — one `trunk`,
    /// driven directly — so a single-graph model needs none of this.
    StageBuilder stage(std::string name, Invocation invocation);

    /// Return to a stage declared earlier, to add a port or a launch descriptor. Throws if no such
    /// stage was declared — `stage()` is what creates one, and silently creating a second here
    /// would put the same name in the order twice.
    StageBuilder edit_stage(std::string name);
    PackageBuilder& metric(std::string name, double value);

    /// The tensors declared so far. Lets a producer check its exported graphs against what the
    /// package promises before anything is written.
    const std::vector<TensorDescriptor>& get_declared_tensors() const noexcept { return package_.io; }
    std::size_t get_graph_count() const noexcept;

    /// Validate and hand over the package. Everything `Package::validate` rejects is something that
    /// would have failed at load; catching it here means a package that cannot run is never written
    /// to disk.
    Package build() const;

private:
    friend class StageBuilder;
    Package package_;
};

}  // namespace RadFiled3D::nn::deploy
