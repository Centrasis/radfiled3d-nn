#include <RadFiled3D/nn/deploy/builder.hpp>

#include <RadFiled3D/nn/exception.hpp>

namespace RadFiled3D::nn::deploy {

TensorBuilder& TensorBuilder::unit(std::string unit) {
    descriptor_.unit = std::move(unit);
    return *this;
}
TensorBuilder& TensorBuilder::range(Range range) {
    descriptor_.range = std::move(range);
    return *this;
}
TensorBuilder& TensorBuilder::normalizer(Normalizer normalizer) {
    descriptor_.normalizer = std::move(normalizer);
    return *this;
}
TensorBuilder& TensorBuilder::dtype(DType dtype) {
    descriptor_.dtype = dtype;
    return *this;
}
TensorBuilder& TensorBuilder::optional(bool optional) {
    descriptor_.optional = optional;
    return *this;
}
PackageBuilder& TensorBuilder::done() { return builder_.descriptor(std::move(descriptor_)); }

PackageBuilder& PackageBuilder::provenance(std::string dataset, std::string software, std::string physics) {
    package_.provenance.dataset = std::move(dataset);
    package_.provenance.software = std::move(software);
    package_.provenance.physics = std::move(physics);
    return *this;
}

PackageBuilder& PackageBuilder::created(std::string timestamp) {
    package_.provenance.created = std::move(timestamp);
    return *this;
}

PackageBuilder& PackageBuilder::field_dimensions_m(std::array<float, 3> dims) {
    package_.geometry.field_dimensions_m = dims;
    return *this;
}

PackageBuilder& PackageBuilder::voxelization(std::array<std::uint32_t, 3> voxel_counts,
                                             std::array<float, 3> voxel_dimensions_m) {
    package_.geometry.voxelization = Voxelization{voxel_counts, voxel_dimensions_m};
    return *this;
}

PackageBuilder& PackageBuilder::field_geometry(const CartesianFieldGeometry& geom)
{
    package_.geometry.field_dimensions_m = {
        geom.get_field_dimensions().x,
        geom.get_field_dimensions().y,
        geom.get_field_dimensions().z
    };
    package_.geometry.voxelization = Voxelization{
        {
            geom.get_voxel_counts().x,
            geom.get_voxel_counts().y,
            geom.get_voxel_counts().z
        },
        geom.get_voxel_dimensions_m()
    };
    return *this;
}

TensorBuilder PackageBuilder::input(std::string name, Semantic semantic, std::vector<std::uint32_t> shape) {
    TensorDescriptor d;
    d.name = std::move(name);
    d.role = Role::Input;
    d.semantic = std::move(semantic);
    d.shape = std::move(shape);
    return TensorBuilder(*this, std::move(d));
}

TensorBuilder PackageBuilder::output(std::string name, Semantic semantic, std::vector<std::uint32_t> shape) {
    TensorDescriptor d;
    d.name = std::move(name);
    d.role = Role::Output;
    d.semantic = std::move(semantic);
    d.shape = std::move(shape);
    return TensorBuilder(*this, std::move(d));
}

PackageBuilder& PackageBuilder::descriptor(TensorDescriptor descriptor) {
    package_.io.push_back(std::move(descriptor));
    return *this;
}

PackageBuilder& PackageBuilder::graph(std::string name, bytes onnx) {
    package_.blocks.push_back(Block::onnx(std::move(name), std::move(onnx)));
    return *this;
}

PackageBuilder& PackageBuilder::block(BlockKind kind, std::string name, std::string target_arch, bytes payload) {
    package_.blocks.push_back(Block{std::move(kind), std::move(name), std::move(target_arch), std::move(payload)});
    return *this;
}

PackageBuilder& PackageBuilder::rf3_metadata(Rf3Metadata metadata) {
    package_.rf3_metadata = std::move(metadata);
    return *this;
}

namespace {

/// The composition being assembled, read out of the block the package already carries.
///
/// Kept in the package rather than beside it so that `build()` needs no assembly step and the
/// invariant "there is at most one composition block" holds at every point, not only at the end.
Composition current_composition(const Package& package) {
    if (const auto found = package.get_composition()) return *found;
    return {};
}

void store_composition(Package& package, const Composition& composition) {
    bytes payload = composition.to_bytes();
    for (auto& b : package.blocks)
        if (b.kind == BlockKind::Composition) {
            b.payload = std::move(payload);
            return;
        }
    package.blocks.push_back(Block{BlockKind::Composition, std::string(kCompositionBlock), {},
                                   std::move(payload)});
}

}  // namespace

PackageBuilder& PackageBuilder::buffer(std::string name, std::uint64_t elements, DType dtype) {
    Composition composition = current_composition(package_);
    composition.buffers.push_back(Buffer{std::move(name), elements, dtype});
    store_composition(package_, composition);
    return *this;
}

StageBuilder PackageBuilder::stage(std::string name, Invocation invocation) {
    Composition composition = current_composition(package_);
    composition.stages.push_back(Stage{name, invocation, {}, {}, {}});
    store_composition(package_, composition);
    return StageBuilder(*this, std::move(name));
}

StageBuilder PackageBuilder::edit_stage(std::string name) {
    const Composition composition = current_composition(package_);
    if (!composition.get_stage(name))
        throw Exception::not_found("composition stage", name);
    return StageBuilder(*this, std::move(name));
}

// ── StageBuilder ──────────────────────────────────────────────────────────────────────────────────
//
// Each call re-reads the stage out of the block, edits it and writes it back, rather than holding a
// pointer into the composition: `store_composition` re-encodes the whole block every time, so a
// pointer taken before one call is dangling after the next.

template <typename F>
StageBuilder& StageBuilder::amend(F&& edit) {
    Composition composition = current_composition(builder_.package_);
    for (auto& s : composition.stages)
        if (s.name == name_) {
            edit(s);
            store_composition(builder_.package_, composition);
            return *this;
        }
    throw Exception::invalid_argument("stage `" + name_ + "` is no longer part of the composition");
}

StageBuilder& StageBuilder::reads(std::string tensor, std::string buffer) {
    return amend([&](Stage& s) { s.reads.push_back(Port{std::move(tensor), std::move(buffer)}); });
}

StageBuilder& StageBuilder::writes(std::string tensor, std::string buffer) {
    return amend([&](Stage& s) { s.writes.push_back(Port{std::move(tensor), std::move(buffer)}); });
}

StageBuilder& StageBuilder::launch(BlockKind kind, std::string target_arch, KernelLaunch launch) {
    return amend([&](Stage& s) {
        s.launches.emplace_back(BlockRef{std::string(kind.get_name()), std::move(target_arch)},
                                std::move(launch));
    });
}

StageBuilder& StageBuilder::weights(bytes payload, std::uint64_t elements, DType dtype,
                                    std::string onnx_external_file) {
    // The block carries the bytes; the stage carries what they mean. An EMPTY buffer stores no
    // block: a zero-length block and an absent one say the same thing, and only one costs bytes.
    const bool has_weights = !payload.empty();
    if (has_weights)
        builder_.package_.blocks.push_back(Block{BlockKind::Weights, name_, {}, std::move(payload)});
    return amend([&](Stage& s) {
        s.weights = Weights{elements, dtype, std::move(onnx_external_file)};
    });
}

PackageBuilder& StageBuilder::done() { return builder_; }

PackageBuilder& PackageBuilder::metric(std::string name, double value) {
    package_.metrics[std::move(name)] = value;
    return *this;
}

std::size_t PackageBuilder::get_graph_count() const noexcept {
    std::size_t n = 0;
    for (const auto& b : package_.blocks)
        if (b.kind == BlockKind::Onnx) ++n;
    return n;
}

Package PackageBuilder::build() const {
    package_.validate();
    return package_;
}

}  // namespace RadFiled3D::nn::deploy
