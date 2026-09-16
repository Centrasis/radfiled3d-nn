#include <RadFiled3D/nn/core/stage_weights.hpp>

namespace RadFiled3D::nn {

StageWeights::StageWeights(std::string stage, const deploy::Package& package) : stage_(std::move(stage)) {
    const deploy::byte_view payload = package.get_weights(stage_);
    bytes_.assign(payload.begin(), payload.end());

    if (const auto composition = package.get_composition())
        if (const auto* declared = composition->get_stage(stage_)) {
            elements_ = declared->weights.elements;
            dtype_ = declared->weights.dtype;
            onnx_external_file_ = declared->weights.onnx_external_file;
        }

    // Always a reference, even for an empty buffer: an implementation asks for its weights the same
    // way whether or not it has any, and a null here would put a branch in every one of them.
    memory_ = memory::host::MemoryRef::of(
        std::span<float>(reinterpret_cast<float*>(bytes_.data()), bytes_.size() / sizeof(float)));
}

deploy::byte_view StageWeights::get_bytes() const noexcept { return deploy::byte_view(bytes_); }

std::vector<StageWeights> load_stage_weights(const deploy::Package& package) {
    std::vector<StageWeights> out;
    if (const auto composition = package.get_composition())
        for (const auto& stage : composition->stages) out.emplace_back(stage.name, package);
    else
        out.emplace_back(std::string(deploy::kTrunkGraph), package);
    return out;
}

}  // namespace RadFiled3D::nn
