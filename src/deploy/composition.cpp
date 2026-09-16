#include <RadFiled3D/nn/deploy/composition.hpp>

// For `kTrunkGraph` and `BlockKind`. The header cannot include this one — package.hpp includes
// composition.hpp — but the translation unit can, and the trunk's role in the order is a
// package-level rule.
#include <RadFiled3D/nn/deploy/package.hpp>
#include <RadFiled3D/nn/exception.hpp>

#include <algorithm>
#include <map>
#include <set>

namespace RadFiled3D::nn::deploy {

std::string_view to_string(Invocation invocation) noexcept {
    switch (invocation) {
        case Invocation::Once: return "once";
        case Invocation::PerQuery: return "per_query";
    }
    return "unknown";
}

Invocation invocation_from_name(std::string_view name) {
    if (name == "once") return Invocation::Once;
    if (name == "per_query") return Invocation::PerQuery;
    throw Exception::invalid_argument("unknown invocation `" + std::string(name) +
                                      "`; expected `once` or `per_query`");
}

// ── the wire form ─────────────────────────────────────────────────────────────────────────────────
//
// Names, not indices. An index into the block list would break the moment a rewrite reordered the
// blocks, and rule 5 explicitly allows a tool to carry blocks it does not understand — which it
// could easily write back in a different order.

void Buffer::encode(Writer& w) const {
    w.string(name);
    w.u64(elements);
    deploy::encode(w, dtype);
}

Buffer Buffer::decode(Reader& r) {
    Buffer b;
    b.name = r.string("buffer name");
    b.elements = r.u64("buffer elements");
    b.dtype = decode_dtype(r, "buffer dtype");
    return b;
}

void Weights::encode(Writer& w) const {
    w.u64(elements);
    deploy::encode(w, dtype);
    w.string(onnx_external_file);
}

Weights Weights::decode(Reader& r) {
    Weights weights;
    weights.elements = r.u64("weights elements");
    weights.dtype = decode_dtype(r, "weights dtype");
    weights.onnx_external_file = r.string("weights onnx external file");
    return weights;
}

void Port::encode(Writer& w) const {
    w.string(tensor);
    w.string(buffer);
}

Port Port::decode(Reader& r) {
    Port p;
    p.tensor = r.string("port tensor");
    p.buffer = r.string("port buffer");
    return p;
}

void BlockRef::encode(Writer& w) const {
    w.string(kind);
    w.string(target_arch);
}

BlockRef BlockRef::decode(Reader& r) {
    BlockRef b;
    b.kind = r.string("launch block kind");
    b.target_arch = r.string("launch target arch");
    return b;
}

void KernelLaunch::encode(Writer& w) const {
    w.string(entry_point);
    for (const auto v : block_size) w.u32(v);
    for (const auto v : grid_size) w.u32(v);
    w.u32(shared_memory_bytes);
}

KernelLaunch KernelLaunch::decode(Reader& r) {
    KernelLaunch k;
    k.entry_point = r.string("launch entry point");
    for (auto& v : k.block_size) v = r.u32("launch block size");
    for (auto& v : k.grid_size) v = r.u32("launch grid size");
    k.shared_memory_bytes = r.u32("launch shared memory");
    return k;
}

std::array<std::uint32_t, 3> KernelLaunch::get_grid_for(std::uint64_t rows) const noexcept {
    std::array<std::uint32_t, 3> grid = grid_size;
    // A zero component means "cover the work", which only the runtime knows the extent of. x spans
    // the rows; y and z default to one plane, because a kernel that wanted more would have said so.
    if (grid[0] == 0) {
        const std::uint64_t block = block_size[0] == 0 ? 1 : block_size[0];
        grid[0] = static_cast<std::uint32_t>((rows + block - 1) / block);
    }
    if (grid[1] == 0) grid[1] = 1;
    if (grid[2] == 0) grid[2] = 1;
    return grid;
}

void Stage::encode(Writer& w) const {
    w.string(name);
    w.string(to_string(invocation));
    w.list(reads, [](Writer& w, const Port& p) { p.encode(w); });
    w.list(writes, [](Writer& w, const Port& p) { p.encode(w); });
    w.list(launches, [](Writer& w, const std::pair<BlockRef, KernelLaunch>& l) {
        l.first.encode(w);
        l.second.encode(w);
    });
    weights.encode(w);
}

Stage Stage::decode(Reader& r) {
    Stage s;
    s.name = r.string("stage name");
    s.invocation = invocation_from_name(r.string("stage invocation"));
    s.reads = r.list("stage reads", [](Reader& r) { return Port::decode(r); });
    s.writes = r.list("stage writes", [](Reader& r) { return Port::decode(r); });
    s.launches = r.list("stage launches", [](Reader& r) {
        BlockRef ref = BlockRef::decode(r);
        KernelLaunch launch = KernelLaunch::decode(r);
        return std::pair<BlockRef, KernelLaunch>{std::move(ref), std::move(launch)};
    });
    s.weights = Weights::decode(r);
    return s;
}

void Composition::encode(Writer& w) const {
    w.list(buffers, [](Writer& w, const Buffer& b) { b.encode(w); });
    w.list(stages, [](Writer& w, const Stage& s) { s.encode(w); });
}

Composition Composition::decode(Reader& r) {
    Composition c;
    c.buffers = r.list("buffers", [](Reader& r) { return Buffer::decode(r); });
    c.stages = r.list("stages", [](Reader& r) { return Stage::decode(r); });
    return c;
}

bytes Composition::to_bytes() const {
    Writer w;
    encode(w);
    return w.take();
}

Composition Composition::from_bytes(byte_view bytes) {
    Reader r(bytes);
    return decode(r);
}

// ── queries ───────────────────────────────────────────────────────────────────────────────────────

const KernelLaunch* Stage::get_launch_for(std::string_view kind, std::string_view arch) const noexcept {
    // An exact architecture match wins over a portable one, for the same reason `select_block` makes
    // that choice: the exact form was compiled for this device and the portable one still has to be
    // JIT-compiled by the driver.
    const KernelLaunch* portable = nullptr;
    for (const auto& [ref, launch] : launches) {
        if (ref.kind != kind) continue;
        if (ref.target_arch == arch) return &launch;
        if (ref.target_arch.empty()) portable = &launch;
    }
    return portable;
}

const Buffer* Composition::get_buffer(std::string_view name) const noexcept {
    for (const auto& b : buffers)
        if (b.name == name) return &b;
    return nullptr;
}

const Stage* Composition::get_stage(std::string_view name) const noexcept {
    for (const auto& s : stages)
        if (s.name == name) return &s;
    return nullptr;
}

const Stage* Composition::get_writer_of(std::string_view buffer) const noexcept {
    for (const auto& s : stages)
        for (const auto& p : s.writes)
            if (p.buffer == buffer) return &s;
    return nullptr;
}

int Composition::get_order_of(std::string_view stage) const noexcept {
    for (std::size_t i = 0; i < stages.size(); ++i)
        if (stages[i].name == stage) return static_cast<int>(i);
    return -1;
}

// ── validation ────────────────────────────────────────────────────────────────────────────────────

void Composition::validate(const std::vector<std::string_view>& block_names,
                           const std::vector<std::string_view>& weights_blocks) const {
    if (stages.empty()) throw Exception::invalid_package("composition records no stages");

    // ── the buffers ──────────────────────────────────────────────────────────────────────────────
    std::set<std::string_view> buffer_names;
    for (const auto& b : buffers) {
        if (b.name.empty()) throw Exception::invalid_package("composition declares an unnamed buffer");
        if (!buffer_names.insert(b.name).second)
            throw Exception::invalid_package("composition declares buffer `" + b.name + "` twice");
        // Zero elements is not a degenerate buffer, it is an unfinished declaration: a stage cannot
        // be wired to memory whose extent nothing states, and a kernel stage has no graph to ask.
        if (b.elements == 0)
            throw Exception::invalid_package("composition buffer `" + b.name + "` declares no elements");
    }

    // ── the stages ───────────────────────────────────────────────────────────────────────────────
    std::set<std::string_view> stage_names;
    for (const auto& s : stages) {
        // Any EXECUTABLE block may implement a stage — an ONNX graph, a PTX blob, an HSACO object.
        // Which one actually runs is `Package::select_block`'s decision, made against the present
        // device; a package is valid as long as something in it carries the stage's name, and it is
        // `Package::require_runnable_on` that decides whether THIS machine can run it.
        if (std::find(block_names.begin(), block_names.end(), s.name) == block_names.end())
            throw Exception::invalid_package("composition stage `" + s.name +
                                             "` names no executable block in this package");
        if (!stage_names.insert(s.name).second)
            throw Exception::invalid_package("composition runs stage `" + s.name + "` twice");

        std::set<std::string_view> tensors;
        for (const auto* ports : {&s.reads, &s.writes})
            for (const auto& p : *ports) {
                if (!tensors.insert(p.tensor).second)
                    throw Exception::invalid_package("stage `" + s.name + "` binds its tensor `" +
                                                     p.tensor + "` to more than one buffer");
                if (!buffer_names.contains(p.buffer))
                    throw Exception::invalid_package("stage `" + s.name + "` names buffer `" + p.buffer +
                                                     "`, which the composition does not declare");
            }

        // The parameters must actually be in the package. An `elements` count with no block behind
        // it is a package that describes weights it does not carry, which fails at load on a device
        // the user cannot debug rather than here, where the producer can still fix it.
        if (s.weights.elements > 0 &&
            std::find(weights_blocks.begin(), weights_blocks.end(), s.name) == weights_blocks.end())
            throw Exception::invalid_package("stage `" + s.name + "` declares " +
                                             std::to_string(s.weights.elements) +
                                             " weights, but the package carries no `weights` block "
                                             "named `" + s.name + "` to hold them");

        for (const auto& [ref, launch] : s.launches) {
            if (launch.entry_point.empty())
                throw Exception::invalid_package("stage `" + s.name + "` declares a " + ref.kind +
                                                 " launch with no entry point");
            // A graph carries its own calling convention; ONNX Runtime is asked for it by name. A
            // launch descriptor beside one would be a second, unreconcilable answer to the same
            // question. An unknown kind is NOT refused — rule 4: it round-trips and is never chosen.
            if (BlockKind::from_name(ref.kind).kind == BlockKind::Onnx)
                throw Exception::invalid_package("stage `" + s.name +
                                                 "` declares a launch descriptor for `onnx`; a graph "
                                                 "carries its own calling convention");
        }
    }

    // The trunk is what produces the model's results, so it must be the last thing that runs. A
    // composition that put it earlier would be describing a model whose outputs feed something else,
    // which the runtime has no way to express and no package has ever meant.
    if (stages.back().name != kTrunkGraph)
        throw Exception::invalid_package("composition ends with `" + stages.back().name + "`; the `" +
                                         std::string(kTrunkGraph) + "` must run last");
    if (stages.back().invocation != Invocation::PerQuery)
        throw Exception::invalid_package("the `" + std::string(kTrunkGraph) +
                                         "` stage must be `per_query`; it is evaluated at every voxel");

    // ── the dataflow ─────────────────────────────────────────────────────────────────────────────
    //
    // One writer per buffer, and every read strictly after that writer. Execution order is recorded
    // rather than inferred, so this is a scan rather than a graph traversal — and expressing a cycle
    // is impossible by construction, which leaves only the two ways a recorded order can be wrong.
    std::map<std::string_view, std::string_view> written_by;
    for (const auto& s : stages)
        for (const auto& p : s.writes)
            if (const auto [it, fresh] = written_by.emplace(p.buffer, s.name); !fresh)
                throw Exception::invalid_package("buffer `" + p.buffer + "` is written by both `" +
                                                 std::string(it->second) + "` and `" + s.name +
                                                 "`; a buffer has one producer");

    std::set<std::string_view> read_buffers;
    for (const auto& s : stages) {
        const int consumer = get_order_of(s.name);
        for (const auto& p : s.reads) {
            read_buffers.insert(p.buffer);
            const auto writer = written_by.find(p.buffer);
            if (writer == written_by.end())
                throw Exception::invalid_package("stage `" + s.name + "` reads buffer `" + p.buffer +
                                                 "`, which no stage writes");
            const int producer = get_order_of(writer->second);
            // THE ordering rule. A buffer holds a value only once its producer has run, so a stage
            // may read what an EARLIER stage wrote and nothing else. Reading its own output, or one
            // a later stage produces, would consume whatever the last inference happened to leave
            // there — a plausible number from the wrong query.
            if (producer >= consumer)
                throw Exception::invalid_package(
                    "stage `" + s.name + "` (position " + std::to_string(consumer) + ") reads buffer `" +
                    p.buffer + "`, which is written by `" + std::string(writer->second) + "` at position " +
                    std::to_string(producer) + "; a stage may only read what an earlier stage wrote");
            // A PerQuery producer feeding a Once consumer would mean one row of the consumer per
            // query, which is not "once" at all. The reverse — Once into PerQuery — is the whole
            // point, and is satisfied by reusing the single row for every query.
            if (stages[static_cast<std::size_t>(producer)].invocation == Invocation::PerQuery &&
                s.invocation == Invocation::Once)
                throw Exception::invalid_package("stage `" + s.name + "` runs once but reads buffer `" +
                                                 p.buffer + "`, which `" + std::string(writer->second) +
                                                 "` writes per query");
        }
    }

    // A buffer nothing reads is memory the runtime would allocate and keep alive for the life of the
    // model to no end. A stage output the CALLER wants is not this: it carries no port at all and is
    // bound directly, so the only way to reach here is to have wired a producer to nothing.
    for (const auto& b : buffers) {
        if (read_buffers.contains(b.name)) continue;
        const auto writer = written_by.find(b.name);
        throw Exception::invalid_package(
            "buffer `" + b.name + "` is " +
            (writer == written_by.end() ? "neither written nor read"
                                        : "written by `" + std::string(writer->second) + "` and never read") +
            "; an intermediate buffer exists to carry a value between stages");
    }
}

}  // namespace RadFiled3D::nn::deploy
