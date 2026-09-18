#include <RadFiled3D/nn/core/model.hpp>

#include <RadFiled3D/nn/exception.hpp>

namespace RadFiled3D::nn {

Model::Model(deploy::Package package) : package_(std::move(package)) {
    // Both resolved ONCE, here, rather than in each backend's constructor. The weights in
    // particular must be: `StageWeights` owns the bytes an ONNX graph's external initializers point
    // at, so their lifetime has to be the model's and not a temporary's.
    composition_ = package_.get_composition();
    weights_ = load_stage_weights(package_);
}

const StageWeights& Model::get_stage_weights(std::string_view stage) const {
    for (const auto& w : weights_)
        if (w.get_stage() == stage) return w;
    throw Exception::not_found("stage", stage);
}

}  // namespace RadFiled3D::nn
