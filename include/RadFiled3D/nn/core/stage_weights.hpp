// A stage's parameters, and the one way every implementation of it receives them.
//
// A stage may be evaluated by an ONNX graph, by a CUDA kernel, by a SPIR-V kernel, or by all three.
// They are three ways to compute the SAME function, so they read ONE set of numbers: the package
// stores the parameters once, under a `weights` block sharing the stage's name, and this is what
// hands them to whichever form actually runs.
//
// The delivery differs by code form, and `docs/custom-code.md` is the contract:
//
//   * **ONNX** — a graph exported with external data records the NAME of a file to find its
//     initializers in. `Weights::onnx_external_file` is that name, and the runtime satisfies it from
//     the package in memory (`AddExternalInitializersFromFilesInMemory`), so nothing has to sit on
//     disk beside the `.rf3m`. A graph that embeds its own initializers records no name and needs
//     none of this.
//   * **A kernel** (PTX, cubin, HSACO, SPIR-V) — the buffer is passed as the kernel's FIRST
//     argument, ahead of the stage's inputs and outputs. That convention is the whole reason this
//     type exists: a blob of machine code has no way to ask where its weights are.
//
// EVERY stage has this, and it may be EMPTY. A stage with no parameters — an activation, a
// reduction, a gather — gets a zero-length buffer rather than nothing, so an implementation reads
// its weights the same way whether or not it has any, and nothing has to branch on their absence.
#pragma once

#include <RadFiled3D/nn/deploy/package.hpp>
#include <RadFiled3D/nn/memory/memory_ref.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace RadFiled3D::nn {

/// One stage's parameters, resident for as long as the model is loaded.
///
/// Materialised ONCE, when the session is built, and never rebuilt — not on a grid change, not per
/// inference. A grid chooses how many queries run; it does not change what the model weighs.
class StageWeights {
public:
    StageWeights(std::string stage, const deploy::Package& package);

    std::string_view get_stage() const noexcept { return stage_; }

    /// The parameters. Empty for a stage that has none.
    deploy::byte_view get_bytes() const noexcept;
    /// How many values they are, as the package declares — so a consumer knows the layout without
    /// parsing the payload.
    std::uint64_t get_elements() const noexcept { return elements_; }
    deploy::DType get_dtype() const noexcept { return dtype_; }

    /// The same memory as a `MemoryRef`, which is how it reaches an implementation that takes a
    /// pointer. Host memory today: mirroring it onto the device is the same gap as the composition's
    /// intermediates (A10), and a kernel stage cannot be launched yet regardless.
    const std::shared_ptr<memory::MemoryRef>& get_memory() const noexcept { return memory_; }

    /// The external-data file name an ONNX implementation refers to these by, or empty when the
    /// graph embeds its own initializers.
    std::string_view get_onnx_external_file() const noexcept { return onnx_external_file_; }

    bool is_empty() const noexcept { return bytes_.empty(); }

private:
    std::string stage_;
    /// Owned rather than a view into the package: a session outlives the bytes it was built from
    /// only if it holds them, and ORT keeps raw pointers into whatever it is given.
    deploy::bytes bytes_;
    std::uint64_t elements_ = 0;
    deploy::DType dtype_ = deploy::DType::F32;
    std::string onnx_external_file_;
    std::shared_ptr<memory::MemoryRef> memory_;
};

/// Materialise every stage's parameters from a package, in execution order.
///
/// A package with no composition has one stage — the `trunk` — so a single-graph model is not a
/// special case here either.
std::vector<StageWeights> load_stage_weights(const deploy::Package& package);

}  // namespace RadFiled3D::nn
