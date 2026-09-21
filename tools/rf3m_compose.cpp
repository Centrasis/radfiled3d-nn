// Add a `Composition` to a package that ships several graphs but no wiring.
//
// The packages the original training framework exported carry `beam_encoder`, `encoding_config` and
// `trunk` as separate ONNX blocks and nothing that says how they connect. The runtime therefore
// treats such a package as a single trunk, the trunk's real inputs (`latent`, `region_state`) are
// left to the caller, and the declared interface (`beam_direction`, `tube_spectrum`) matches
// nothing — the gap `FieldInference.TheV1InterfaceGapIsReportedAndNotSilent` pins.
//
// This writes the missing block. Afterwards the runtime runs the encoders itself, carries their
// results in device memory, and a caller can skip a stage to reuse what it produced last time
// (`set_stage_enabled`) — so a beam that has not moved costs nothing and a grid that has not
// changed costs nothing.
//
// THE WIRING IS DISCOVERED, NOT ASSUMED. Each graph is opened and asked what it consumes and
// produces; a producer's output feeds a consumer's input when the names match, or failing that when
// exactly one unfed input has the same width. Anything ambiguous is reported and nothing is
// written — a wrong composition would be worse than none, because the runtime would believe it.
#include <RadFiled3D/nn.hpp>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace RadFiled3D::nn;

namespace {

struct Tensor {
    std::string name;
    std::int64_t width = 0;      ///< elements per row; a symbolic batch axis counts as one row
    bool batched = false;        ///< leading axis is symbolic, i.e. one row per query
};

struct Graph {
    std::string name;
    std::vector<Tensor> inputs, outputs;
};

/// What a graph consumes and produces, asked of the runtime rather than parsed.
Graph describe(Ort::Env& env, const std::string& name, deploy::byte_view onnx) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    Ort::Session session(env, onnx.data(), onnx.size(), options);
    Ort::AllocatorWithDefaultOptions allocator;

    const auto collect = [&](bool is_input) {
        std::vector<Tensor> out;
        const std::size_t count = is_input ? session.GetInputCount() : session.GetOutputCount();
        for (std::size_t i = 0; i < count; ++i) {
            Tensor t;
            t.name = is_input ? session.GetInputNameAllocated(i, allocator).get()
                              : session.GetOutputNameAllocated(i, allocator).get();
            const auto info = is_input ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);
            const auto shape = info.GetTensorTypeAndShapeInfo().GetShape();
            t.width = 1;
            for (std::size_t d = 0; d < shape.size(); ++d) {
                if (d == 0 && shape[d] < 0) {   // symbolic leading axis: the batch
                    t.batched = true;
                    continue;
                }
                t.width *= shape[d] > 0 ? shape[d] : 1;
            }
            out.push_back(std::move(t));
        }
        return out;
    };
    return Graph{name, collect(true), collect(false)};
}

[[noreturn]] void die(const std::string& what) {
    std::cerr << "rf3m_compose: " << what << "\n";
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 3) {
        std::cerr << "usage: rf3m_compose <in.rf3m> <out.rf3m> [--trunk NAME]\n";
        return 2;
    }
    const std::string in = argv[1], out = argv[2];
    std::string trunk_name = "trunk";
    for (int i = 3; i < argc; ++i)
        if (std::string(argv[i]) == "--trunk" && i + 1 < argc) trunk_name = argv[++i];
    if (!std::filesystem::exists(in)) die("no package at " + in);

    const deploy::Package package = deploy::Package::read_file(in);
    if (package.get_composition())
        die(in + " already carries a composition; nothing to do");

    // ── what each graph consumes and produces ───────────────────────────────────────────────
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "rf3m_compose");
    std::vector<Graph> graphs;
    for (const auto& block : package.blocks)
        if (block.kind == deploy::BlockKind::Onnx)
            graphs.push_back(describe(env, block.name, deploy::byte_view(block.payload)));
    if (graphs.size() < 2) die("this package has fewer than two graphs; it needs no composition");

    const auto trunk = std::find_if(graphs.begin(), graphs.end(),
                                    [&](const Graph& g) { return g.name == trunk_name; });
    if (trunk == graphs.end()) die("no graph named `" + trunk_name + "` to run last");

    // ── match producers to consumers ────────────────────────────────────────────────────────
    struct Edge {
        std::string producer, produced_tensor;     // stage and its output
        std::string consumer, consumed_tensor;     // stage and its input
        std::int64_t width = 0;
    };
    std::vector<Edge> edges;
    std::set<std::string> fed;   // "stage.tensor" the wiring supplies, so a caller need not

    for (const Graph& producer : graphs) {
        if (producer.name == trunk_name) continue;   // the trunk produces the model's outputs
        for (const Tensor& output : producer.outputs) {
            std::vector<const Tensor*> candidates;
            for (const Tensor& input : trunk->inputs) {
                if (fed.count(trunk->name + "." + input.name)) continue;
                if (input.width != output.width) continue;
                candidates.push_back(&input);
            }
            if (candidates.empty()) continue;
            // An exact name match wins; otherwise the match must be unambiguous.
            const Tensor* chosen = nullptr;
            for (const Tensor* c : candidates)
                if (c->name == output.name) chosen = c;
            if (chosen == nullptr) {
                if (candidates.size() > 1)
                    die("`" + producer.name + "." + output.name + "` (" + std::to_string(output.width) +
                        " wide) could feed several inputs of `" + trunk_name +
                        "`; the wiring is ambiguous and nothing was written");
                chosen = candidates.front();
            }
            fed.insert(trunk->name + "." + chosen->name);
            edges.push_back({producer.name, output.name, trunk->name, chosen->name, output.width});
        }
    }
    if (edges.empty()) die("no output of any graph fits an input of `" + trunk_name + "`");

    // ── add the wiring to the package that was read ─────────────────────────────────────────
    //
    // The parsed package is kept AS IT IS and gains one block. Rebuilding it through
    // `PackageBuilder` would mean restating every tensor through `input()`/`output()`, and anything
    // the descriptors carry that the restatement forgot — a range, a normalizer, a dtype — would be
    // quietly dropped. Appending is the only way to be sure nothing else changed.
    deploy::Package composed = package;
    deploy::Composition composition;

    // One buffer per edge, named after what the trunk calls it.
    for (const Edge& e : edges)
        composition.buffers.push_back(
            deploy::Buffer{e.consumed_tensor, std::uint64_t(e.width), deploy::DType::F32});

    // The producers run ONCE per field: they describe the beam and the region, not a point in
    // space. `rows_for` then gives them a single row, which the runtime repeats across the queries
    // in DEVICE memory — so a caller never fans a latent out by hand, and a producer whose input
    // has not changed can simply be skipped, keeping what it wrote last time.
    std::cout << "composition\n";
    for (const Graph& producer : graphs) {
        if (producer.name == trunk_name) continue;
        deploy::Stage stage;
        stage.name = producer.name;
        stage.invocation = deploy::Invocation::Once;
        for (const Edge& e : edges)
            if (e.producer == producer.name) {
                stage.writes.push_back(deploy::Port{e.produced_tensor, e.consumed_tensor});
                std::cout << "  " << producer.name << "." << e.produced_tensor << "  ->  ["
                          << e.consumed_tensor << "] " << e.width << " wide  ->  " << trunk_name << "."
                          << e.consumed_tensor << "   (once per field)\n";
            }
        composition.stages.push_back(std::move(stage));
    }
    deploy::Stage last;
    last.name = trunk_name;
    last.invocation = deploy::Invocation::PerQuery;
    for (const Edge& e : edges)
        last.reads.push_back(deploy::Port{e.consumed_tensor, e.consumed_tensor});
    composition.stages.push_back(std::move(last));

    composed.blocks.push_back(deploy::Block{deploy::BlockKind::Composition,
                                            std::string(deploy::kCompositionBlock), {},
                                            composition.to_bytes()});
    // The same validation a load performs, so a wiring that could not run is never written: the
    // ordering rule, one writer per buffer, and no buffer nothing reads.
    composed.validate();
    composed.write_file(out);
    std::cout << "wrote       " << out << "\n";
    std::cout << "caller now binds:";
    for (const Tensor& input : trunk->inputs)
        if (!fed.count(trunk->name + "." + input.name)) std::cout << " " << input.name;
    for (const Graph& g : graphs)
        if (g.name != trunk_name)
            for (const Tensor& input : g.inputs) std::cout << " " << input.name;
    std::cout << "\n";
    return 0;
} catch (const std::exception& err) {
    std::cerr << "rf3m_compose: " << err.what() << "\n";
    return 1;
}
