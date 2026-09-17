#include <RadFiled3D/nn/core/field_inference.hpp>

#include <RadFiled3D/nn/memory/memory_ref.hpp>

#include <RadFiled3D/Voxel.hpp>

#include <cmath>
#include <span>
#include <utility>

namespace RadFiled3D::nn {

namespace {

/// A metric value as the graph wants it, or a refusal naming the normalizer.
///
/// `deploy::apply` answers `nullopt` for a normalizer this build does not implement or a degenerate
/// parameterisation, and a plausible-looking number in its place is exactly the silently wrong
/// result the design forbids (R-I9).
float to_graph_units(const deploy::Normalizer& normalizer, double metric, const std::string& tensor) {
    const std::optional<double> converted = deploy::apply(normalizer, metric);
    if (!converted)
        throw Exception::invalid_argument("cannot normalize `" + tensor + "`: this build does not implement `" +
                                          std::string(deploy::get_name(normalizer)) +
                                          "`, or its parameters are degenerate");
    return static_cast<float>(*converted);
}

/// Add a layer of `elements` floats per voxel, or keep a compatible one that already exists.
///
/// One element is a plain scalar layer; more than one is a histogram layer, which is how RadFiled3D
/// represents a per-voxel vector (a spectrum's bins). The two differ only in the voxel template —
/// the element data is contiguous either way, which is what lets the graph write into it directly.
void add_output_layer(VoxelBuffer& buffer, const std::string& name, std::uint64_t elements,
                      const std::string& unit) {
    if (buffer.has_layer(name)) {
        // Keeping a layer of the wrong width would bind a buffer the graph overruns.
        const std::size_t have = buffer.get_layer(name).get_voxel_flat_raw(0)->get_bytes() / sizeof(float);
        if (have != elements)
            throw Exception::invalid_argument("layer `" + name + "` already exists with " + std::to_string(have) +
                                              " elements per voxel, the model produces " + std::to_string(elements));
        return;
    }
    if (elements == 1) {
        buffer.add_layer<float>(name, 0.f, unit);
        return;
    }
    // The template carries a NULL data pointer on purpose. `add_custom_layer` reads only
    // `get_bytes()` from it — the bin count out of the histogram header, never the data — and then
    // allocates the layer's own buffer and points every voxel view into it. Handing it a real
    // buffer would make the template's pointer look meaningful when it is discarded.
    //
    // The bin width stays zero: the descriptor's `Range` carries the energy axis, not this call, and
    // inventing a width would put a number in the file that nothing derived.
    const HistogramVoxel<float> tmpl(static_cast<std::size_t>(elements), 0.f, nullptr);
    buffer.add_custom_layer<HistogramVoxel<float>, float>(name, tmpl, 0.f, unit);
}

}  // namespace

std::vector<float> make_voxel_center_positions(const CartesianFieldGeometry& geometry) {
    const auto counts = geometry.voxel_counts;
    const auto voxel = geometry.get_voxel_dimensions_m();
    std::vector<float> positions(geometry.get_voxel_count() * 3);

    // RadFiled3D's flat index is `z * ny * nx + y * nx + x`, so X varies fastest and Z slowest.
    // Writing the loops in that order is what keeps query `q` the position of voxel `q`.
    std::size_t at = 0;
    for (std::uint32_t z = 0; z < counts[2]; ++z)
        for (std::uint32_t y = 0; y < counts[1]; ++y)
            for (std::uint32_t x = 0; x < counts[0]; ++x) {
                // The CENTRE, not the corner: a voxel is a cell, and sampling its corner would bias
                // the whole field by half a voxel along every axis.
                positions[at++] = (static_cast<float>(x) + 0.5f) * voxel[0];
                positions[at++] = (static_cast<float>(y) + 0.5f) * voxel[1];
                positions[at++] = (static_cast<float>(z) + 0.5f) * voxel[2];
            }
    return positions;
}

FieldInference::FieldInference(std::shared_ptr<InferenceSession> session,
                               std::shared_ptr<GPUCartesianRadiationField> field, std::string channel)
    : session_(std::move(session)), field_(std::move(field)), channel_(std::move(channel)) {
    if (!session_) throw Exception::invalid_argument("FieldInference: null session");
    if (!field_) throw Exception::invalid_argument("FieldInference: null field");

    const deploy::Package& package = session_->get_package();

    kind_ = package.get_model_kind();
    const CartesianFieldGeometry geometry = field_->get_geometry();

    // A whole-volume model's grid is the MODEL's, not the caller's: the architecture fixes it, and a
    // package that records one says so. Filling a field of another shape is not a resize, it is a
    // different field, so the mismatch is refused here rather than discovered as a capacity error
    // against a graph dimension, which names no geometry and reads like a bug in the binding.
    if (kind_ == deploy::ModelKind::WholeVolume && package.geometry.voxelization) {
        const auto& fixed = package.geometry.voxelization->voxel_counts;
        if (fixed != geometry.voxel_counts)
            throw Exception::invalid_argument(
                "this model emits a fixed " + std::to_string(fixed[0]) + "x" + std::to_string(fixed[1]) +
                "x" + std::to_string(fixed[2]) + " volume, but the field is " +
                std::to_string(geometry.voxel_counts[0]) + "x" + std::to_string(geometry.voxel_counts[1]) +
                "x" + std::to_string(geometry.voxel_counts[2]));
    }

    // The query count is the voxel count either way: it is what a voxelwise model is asked for one
    // point at a time, and what resolves a whole-volume graph's dynamic axes.
    session_->set_voxel_grid(geometry.voxel_counts);

    // ── the positions, generated from the grid and never crossing the API (R-I1) ─────────────────
    //
    // Only a voxelwise model takes them. A whole-volume model has no position input at all — its
    // spatial structure is in the graph — so there is nothing here to generate or bind.
    if (kind_ == deploy::ModelKind::VoxelWise) {
        const deploy::TensorDescriptor* position =
            package.get_tensor_with(deploy::Role::Input, deploy::Semantic::Position);
        position_input_ = position->name;
        positions_ = make_voxel_center_positions(geometry);
        if (position->get_elements() != 3)
            throw Exception::invalid_argument("position input `" + position_input_ + "` takes " +
                                              std::to_string(position->get_elements()) +
                                              " elements per query; a Cartesian field supplies 3");
        for (float& coordinate : positions_)
            coordinate = to_graph_units(position->normalizer, coordinate, position_input_);
        session_->bind_input(position_input_, memory::host::MemoryRef::of(std::span<float>(positions_)));
    }

    // ── one layer per declared output ───────────────────────────────────────────────────────────
    //
    // Derived from the descriptors, never from a fixed list of names: a model that grows a new
    // output must cost no change here (R-F2, defect D3). `kFluxLayer` and `kSpectrumLayer` are the
    // names today's models happen to use, not the set this code supports.
    auto buffer = field_->add_channel(channel_);
    for (const deploy::TensorDescriptor* output : package.get_outputs()) {
        if (output->dtype != deploy::DType::F32)
            throw Exception::invalid_argument("output `" + output->name + "` is " +
                                              std::string(deploy::to_string(output->dtype)) +
                                              "; a field layer holds float32");
        // A field layer is labelled with the descriptor's unit, so the numbers in it must be METRIC.
        // A normalized output is therefore inverted after the run; one whose normalizer this build
        // cannot invert is refused here rather than written out under a label it does not match
        // (rule 9), while the package still round-trips and can still be described (rule 4).
        if (!deploy::is_invertible(output->normalizer))
            throw Exception::invalid_argument(
                "output `" + output->name + "` is normalized with `" +
                std::string(deploy::get_name(output->normalizer)) +
                "`, which this build cannot invert; the field's layer would carry graph units under "
                "the label `" + (output->unit.empty() ? std::string("(none)") : output->unit) + "`");
        // A voxelwise model emits `[queries, elements]` — VOXEL-MAJOR, which is exactly how a
        // RadFiled3D layer is laid out, so the graph writes into the layer directly. A whole-volume
        // model's multi-element output is typically CHANNEL-major (`[1, C, D, H, W]`: all of channel
        // 0 over the grid, then channel 1), and the container records NEITHER which of the two it is
        // NOR how the spatial axes map to `z*ny*nx + y*nx + x`. Binding the layer anyway would write
        // a transposed field that passes every finite-and-nonzero check there is.
        //
        // So it is refused by name until the format carries the layout. A single-element output has
        // no channel axis to transpose and is unaffected, which is why `flux` still works.
        if (kind_ == deploy::ModelKind::WholeVolume && output->get_elements() > 1)
            throw Exception::invalid_argument(
                "output `" + output->name + "` has " + std::to_string(output->get_elements()) +
                " elements per voxel and this model emits a whole volume; the container does not "
                "record whether such an output is channel-major or voxel-major, so writing it into "
                "a layer would risk a transposed field. Bind it yourself and place the values, or "
                "use a model queried per voxel");
        add_output_layer(*buffer, output->name, output->get_elements(), output->unit);
        output_layers_.push_back(output->name);
        if (!std::holds_alternative<deploy::Identity>(output->normalizer))
            normalized_outputs_.emplace_back(output->name, output->normalizer);
    }

    // Bind each layer's own storage. A layer with a GPU mirror attached is bound INSTEAD, so a
    // device-resident result never round-trips through the host — at the cost that the host voxels
    // then stay whatever they were, which `run()` says out loud.
    for (const std::string& layer : output_layers_) {
        if (auto mirror = field_->get_device_memory(channel_, layer)) {
            // Inverting a normalizer is a HOST transform over the result, and there is no kernel
            // here to run one on device memory (the same constraint as A6 on the input side). A
            // mirror plus a normalized output would leave graph units in the renderer's buffer with
            // nothing to say so, so the combination is refused rather than half-honoured.
            if (is_normalized(layer))
                throw Exception::invalid_argument(
                    "output `" + layer + "` is normalized and its layer has device memory attached; "
                    "inverting a normalizer is a host transform, so bind a host layer or invert on "
                    "the device yourself");
            session_->bind_output(layer, std::move(mirror));
            continue;
        }
        const std::uint64_t floats = buffer->get_voxel_count() *
                                     (buffer->get_layer(layer).get_voxel_flat_raw(0)->get_bytes() / sizeof(float));
        session_->bind_output(layer,
                              memory::host::MemoryRef::of(std::span<float>(buffer->get_layer<float>(layer),
                                                                           static_cast<std::size_t>(floats))));
    }

    // What the caller still owes. Reported as a list rather than discovered one failed run at a
    // time, and deliberately not enforced here: the caller binds between construction and `run()`.
    for (const deploy::TensorDescriptor* input : package.get_inputs())
        if (input->name != position_input_ && !input->optional) caller_inputs_.push_back(input->name);
}

bool FieldInference::is_normalized(const std::string& layer) const noexcept {
    for (const auto& [name, normalizer] : normalized_outputs_)
        if (name == layer) return true;
    return false;
}

void FieldInference::run() {
    session_->infer();

    // The graph emitted its own units; the field's layers are labelled with the descriptor's, so
    // every normalized output is converted back in place. Done after the run rather than on read,
    // because the layer IS the result: a caller reaches for `get_layer<float>()`, or hands the field
    // to FieldStore, and neither goes through anything this class could intercept.
    auto buffer = field_->get_channel(channel_);
    for (const auto& [layer, normalizer] : normalized_outputs_) {
        const std::size_t elements =
            buffer->get_layer(layer).get_voxel_flat_raw(0)->get_bytes() / sizeof(float);
        float* values = buffer->get_layer<float>(layer);
        const std::size_t count = buffer->get_voxel_count() * elements;
        for (std::size_t i = 0; i < count; ++i) {
            const std::optional<double> metric = deploy::invert(normalizer, values[i]);
            if (!metric)
                // Reached when the network emitted a value the transform cannot carry back — an
                // exponent that overflows, say. Naming the voxel is what makes that diagnosable;
                // leaving the number in place would put a plausible-looking one in a stored field.
                throw Exception::invalid_argument(
                    "output `" + layer + "` element " + std::to_string(i) + " is " +
                    std::to_string(values[i]) + ", which `" + std::string(deploy::get_name(normalizer)) +
                    "` cannot invert to a finite metric value");
            values[i] = static_cast<float>(*metric);
        }
    }
}

}  // namespace RadFiled3D::nn
