// Filling a radiation field by running a session over its voxel grid.
//
// This is the join between the two halves of the library: `InferenceSession` knows how to run a
// model against caller-owned buffers, `GPUCartesianRadiationField` knows how to hold a field, and
// nothing so far connected them. A caller had to derive the query count from the grid, generate one
// position per voxel, add a layer per output and bind each one — a dozen lines that must agree with
// the field's voxel index order, and are wrong in a way no type can catch if they do not.
//
// Two facts about RadFiled3D make the zero-copy form possible, and both were verified rather than
// assumed:
//
//   * A layer keeps its voxel VIEWS and its element DATA in two separate buffers, so
//     `get_layer<float>()` is a contiguous `voxel_count * elements_per_voxel` run even for a
//     multi-bin histogram layer. The graph writes into the field itself; nothing is staged.
//   * The flat voxel index is `z * ny * nx + y * nx + x` (`VoxelGrid::get_voxel_idx`), so X varies
//     fastest. Generated positions follow that order exactly — a mismatch would produce a field
//     that looks plausible and is transposed.
#pragma once

#include <RadFiled3D/nn/core/field.hpp>
#include <RadFiled3D/nn/core/session.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace RadFiled3D::nn {

/// Runs a session over every voxel of a field and writes each declared output into a layer.
///
/// Construction prepares everything derivable from the field and the package; `run()` executes.
/// The split exists because a model's remaining inputs — a beam direction, a latent code, a tube
/// spectrum — are the CALLER's to supply, and they are bound on the session between the two calls:
///
///     FieldInference filler(session, field);
///     session->bind_input("beam_direction", direction);
///     filler.run();
///
/// **That order is load-bearing.** Construction calls `set_voxel_grid`, which discards every
/// existing binding — the bound shapes depend on the query count, so a grid change cannot keep
/// them. A buffer bound BEFORE the constructor is therefore silently gone; `run()` then reports it
/// as missing, which is the failure, not the cause.
///
/// **This object owns the bindings' lifetime.** ORT keeps the raw pointer of every bound tensor, so
/// the generated positions live in a member buffer and the field is held by `shared_ptr` — the
/// output bindings point into its layers. Destroying either while the session can still run would
/// leave the session reading freed memory, which is why neither is borrowed.
class FieldInference {
public:
    /// Prepare `session` to fill `field`.
    ///
    /// Sets the voxel grid from the field's geometry, adds one layer per declared output (creating
    /// `channel` if needed) and binds every one. What happens on the input side depends on which
    /// kind of model the package carries, and the class handles both:
    ///
    ///   * **`VoxelWise`** — an FCNN over a coordinate (PBRFNet, TPBRFNet). One position per voxel is
    ///     generated, normalized and bound; the grid is the caller's to choose.
    ///   * **`WholeVolume`** — a CNN emitting the grid in one run. There is no position input to
    ///     supply. The grid belongs to the MODEL, so a field whose shape disagrees with the
    ///     package's recorded voxelization is refused rather than resized.
    ///
    /// A `WholeVolume` model's output of more than one element per voxel is refused: the container
    /// records neither whether such an output is channel-major or voxel-major nor how its spatial
    /// axes map to RadFiled3D's flat index, and guessing would produce a transposed field that looks
    /// entirely plausible.
    FieldInference(std::shared_ptr<InferenceSession> session,
                   std::shared_ptr<GPUCartesianRadiationField> field,
                   std::string channel = std::string(kPredictionChannel));

    /// Run the model. Afterwards the field's layers hold the prediction IN METRIC UNITS.
    ///
    /// Inputs the caller never bound are reported by the session, all at once (R-I1).
    ///
    /// A normalized output is inverted in place once the run completes, so a layer labelled `eV`
    /// holds electronvolts and not whatever the network was trained to emit. A graph value the
    /// transform cannot carry back to a finite metric one throws, naming the layer and the element
    /// — a plausible number left in a field that is then stored is not recoverable later.
    void run();

    /// The layers this object created and bound, in declaration order.
    const std::vector<std::string>& get_output_layers() const noexcept { return output_layers_; }
    /// The channel the outputs were written into.
    const std::string& get_channel() const noexcept { return channel_; }
    /// The graph tensor carrying `Semantic::Position`, which this object supplies. Empty for a
    /// `WholeVolume` model, which has none.
    const std::string& get_position_input() const noexcept { return position_input_; }
    /// Which kind of model is being driven — the distinction between an FCNN queried per voxel and a
    /// CNN emitting the whole field.
    deploy::ModelKind get_model_kind() const noexcept { return kind_; }
    /// The package's non-optional inputs other than the position — what the caller is expected to
    /// bind. Fixed at construction: it is what the PACKAGE declares, not what is still unbound, and
    /// it does not shrink as bindings are made. The session answers the live question, naming every
    /// input actually missing when `run()` preflights.
    ///
    /// A name here is not a guarantee it can be bound: a package may declare the interface of a
    /// model whose graph names different tensors, which `bind_input` then refuses by name.
    const std::vector<std::string>& get_caller_inputs() const noexcept { return caller_inputs_; }

    const GPUCartesianRadiationField& get_field() const noexcept { return *field_; }
    std::shared_ptr<GPUCartesianRadiationField> share_field() const noexcept { return field_; }

private:
    std::shared_ptr<InferenceSession> session_;
    std::shared_ptr<GPUCartesianRadiationField> field_;
    std::string channel_;
    deploy::ModelKind kind_ = deploy::ModelKind::VoxelWise;
    std::string position_input_;
    std::vector<std::string> output_layers_;
    std::vector<std::string> caller_inputs_;
    /// The generated voxel-centre positions, in METRIC units already passed through the position
    /// descriptor's normalizer. Held here because the session binds it by pointer.
    std::vector<float> positions_;
    /// Outputs whose values need converting back after a run, with the transform to use. Only the
    /// non-`Identity` ones: an identity pass over a whole volume is measurable work for no change.
    std::vector<std::pair<std::string, deploy::Normalizer>> normalized_outputs_;

    bool is_normalized(const std::string& layer) const noexcept;
};

/// The metric centre of every voxel of `geometry`, in RadFiled3D's flat voxel order (X fastest),
/// as `voxel_count * 3` floats.
///
/// Exposed because it is the one piece of A4 a caller may legitimately want on its own — to feed a
/// model through a path this class does not cover, or to check what was queried.
std::vector<float> make_voxel_center_positions(const FieldGeometry& geometry);

}  // namespace RadFiled3D::nn
