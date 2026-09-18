// `nn::Model` — what a loaded model IS, with no notion of how it will be run.
//
// This is the seam between `rfnn::deploy` and `rfnn::inference`. The container hands over a
// `deploy::Package`, a value describing the model; a concrete backend is handed a `Model` and uses
// it to build its stages and size its buffers. Nothing here executes anything, allocates device
// memory or names an execution provider, which is what lets a backend be written against the model
// rather than against the byte layout.
//
// It is NOT a second copy of the package's API. What it adds is the work every backend otherwise
// repeats in its own constructor: resolving the composition once, and loading the per-stage weights
// once, so those exist exactly as long as the model does. Everything else stays a question for the
// package, reachable through `get_package()` — there is one source of truth for the format and it
// is not this class.
#pragma once

#include <RadFiled3D/nn/core/stage_weights.hpp>
#include <RadFiled3D/nn/deploy/package.hpp>
#include <RadFiled3D/nn/types.hpp>

#include <optional>
#include <string_view>
#include <vector>

namespace RadFiled3D::nn {

class Model {
public:
    /// Virtual, because a backend may specialise this and a `Model` is held by base reference.
    virtual ~Model() = default;

    /// Takes the package by value: the model OWNS its description. A stage's weights are viewed as
    /// bytes inside it (ONNX Runtime keeps the pointer), so a package that outlived only the caller
    /// would leave those views dangling.
    explicit Model(deploy::Package package);

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    /// The package this was built from — still the one source of truth for the format.
    const deploy::Package& get_package() const noexcept { return package_; }

    /// How the stages fit together, or `nullopt` for a package that is a single trunk.
    const std::optional<deploy::Composition>& get_composition() const noexcept { return composition_; }

    /// Every stage's parameters, in the package's order. A stage with none still has an entry, so
    /// nothing has to branch on their absence.
    const std::vector<StageWeights>& get_stage_weights() const noexcept { return weights_; }

    /// One stage's parameters. Throws `not_found` rather than returning an empty set for a stage
    /// the package never declared — an unknown name is a mistake, not an absence of weights.
    const StageWeights& get_stage_weights(std::string_view stage) const;

    /// Queried per position, or emitting the whole volume. DERIVED from the declared interface, so
    /// it can never contradict it.
    deploy::ModelKind get_model_kind() const noexcept { return package_.get_model_kind(); }
    bool is_voxelwise() const noexcept { return package_.is_voxelwise(); }

    /// The grid the model was trained on, when it recorded one (R-F4). `nullopt` is an ordinary
    /// answer: the metric box alone is not a grid.
    std::optional<CartesianFieldGeometry> get_field_geometry() const {
        return package_.geometry.get_field_geometry();
    }

    std::vector<const deploy::TensorDescriptor*> get_inputs() const { return package_.get_inputs(); }
    std::vector<const deploy::TensorDescriptor*> get_outputs() const { return package_.get_outputs(); }

protected:
    deploy::Package package_;
    std::optional<deploy::Composition> composition_;
    std::vector<StageWeights> weights_;
};

}  // namespace RadFiled3D::nn
