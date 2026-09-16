// One executable stage of a loaded model, whatever it is made of.
//
// A model is a chain of stages. Most are ONNX graphs; some are compiled kernels — PTX, a cubin, an
// AMD code object. The session runs them in the recorded order and does NOT care which is which,
// which is what this interface is for: without it, every place that drives a model would branch on
// the stage's kind, and adding a kind would mean editing all of them.
//
// Implementations live with their backend, never here:
//
//     nn::onnx::GraphStage   an ONNX graph, run through ONNX Runtime's IoBinding
//     nn::cuda::KernelStage  PTX or a cubin, launched through the CUDA driver API
//
// and a session obtains one by ASKING (`ComputeBackend::make_kernel_stage`), so `backends/onnx.cpp`
// contains no CUDA include and rule 7b2 keeps holding.
//
// The protocol is the same for both, and it is the reason a repeated inference allocates nothing:
//
//     set_rows(n)                     once per grid — resolves shapes, drops old bindings
//     bind(tensor, side, memory)      once per grid — the memory stays put
//     run()                           per inference — executes and nothing else
//
// `docs/custom-code.md` is the contract a compiled stage is held to, including the argument order a
// kernel receives.
#pragma once

#include <RadFiled3D/nn/deploy/composition.hpp>
#include <RadFiled3D/nn/memory/memory_ref.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace RadFiled3D::nn {

class Stage {
public:
    virtual ~Stage() = default;

    Stage(const Stage&) = delete;
    Stage& operator=(const Stage&) = delete;

    /// Identifies the stage AND names the blocks that implement it.
    virtual std::string_view get_name() const noexcept = 0;
    virtual deploy::Invocation get_invocation() const noexcept = 0;

    /// The tensors this stage declares, in declaration order.
    virtual const std::vector<std::string>& get_inputs() const noexcept = 0;
    virtual const std::vector<std::string>& get_outputs() const noexcept = 0;

    /// How many ELEMENTS one run reads from, or writes to, a tensor at the current row count.
    /// Throws `NotFound` for a tensor this stage does not declare.
    virtual std::uint64_t get_elements(std::string_view tensor, bool is_input) const = 0;

    /// Choose the number of rows one run processes, and drop every binding.
    ///
    /// `rows` is the query count for a `PerQuery` stage and 1 for a `Once` one — the session applies
    /// that rule, because whether a batch axis means "per query" is a fact only the package records.
    virtual void set_rows(std::int64_t rows) = 0;
    virtual std::int64_t get_rows() const noexcept = 0;

    /// Attach memory to one of this stage's tensors. It is read again on every run, so editing the
    /// buffer in place is picked up without rebinding, and the reference must outlive the binding.
    virtual void bind(std::string_view tensor, bool is_input,
                      std::shared_ptr<memory::MemoryRef> memory) = 0;

    /// Hand this stage its parameters, in the backend's own memory domain.
    ///
    /// Called once when the session is built and never again — a grid chooses how many queries run,
    /// not what the model weighs. A GRAPH ignores it: ONNX Runtime already has the initializers,
    /// either embedded or supplied as external data when the session was constructed. A KERNEL needs
    /// it, because the buffer is its first argument and there is no other way to tell it where.
    virtual void set_weights(std::shared_ptr<memory::MemoryRef>) {}

    /// Whether something is attached to a tensor. The session reports EVERY missing binding at once
    /// rather than the first, and this is what it asks.
    virtual bool is_bound(std::string_view tensor, bool is_input) const = 0;

    /// Execute one run. Allocates nothing.
    virtual void run() = 0;

    /// Rows one run produces given a query count: a stage that runs once per field emits a single
    /// row whatever batch axis it declares. `beam_encoder` declares `[batch, 3]` on every input and
    /// still runs once, because a beam is constant over the volume — which is exactly why the
    /// container records the invocation instead of the runtime inferring it from a shape.
    static std::int64_t rows_for(deploy::Invocation invocation, std::int64_t queries) noexcept {
        return invocation == deploy::Invocation::Once ? 1 : queries;
    }

protected:
    Stage() = default;
};

}  // namespace RadFiled3D::nn
