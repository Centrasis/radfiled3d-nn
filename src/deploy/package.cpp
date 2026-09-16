#include <RadFiled3D/nn/deploy/package.hpp>

#include <RadFiled3D/nn/deploy/version.hpp>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <set>

namespace RadFiled3D::nn::deploy {

// ── BlockKind / Block ─────────────────────────────────────────────────────────────────────────────

namespace {
constexpr std::pair<BlockKind::Kind, std::string_view> kBlockKindNames[] = {
    {BlockKind::Onnx, "onnx"},       {BlockKind::CudaPtx, "cuda_ptx"}, {BlockKind::CudaCubin, "cuda_cubin"},
    {BlockKind::Hsaco, "hsaco"},     {BlockKind::OpenCl, "opencl"},  {BlockKind::SpirV, "spirv"},
    {BlockKind::Weights, "weights"},   {BlockKind::Composition, "composition"},
};
}  // namespace

BlockKind BlockKind::from_name(std::string_view name) {
    for (const auto& [kind, known] : kBlockKindNames)
        if (known == name) return BlockKind(kind);
    BlockKind k;
    k.custom_name = std::string(name);
    return k;
}

std::string_view BlockKind::get_name() const noexcept {
    for (const auto& [kind, known] : kBlockKindNames)
        if (kind == this->kind) return known;
    return custom_name;
}

std::string_view BlockKind::get_target() const noexcept {
    switch (kind) {
        case CudaPtx:
        case CudaCubin: return "cuda";
        case Hsaco: return "rocm";
        case OpenCl: return "opencl";
        // No `Backend` answers to either name, so neither is ever selected. That is the truth about
        // this build rather than an oversight: a package may carry the code, and nothing here runs it.
        case SpirV: return "spirv";
        default: return {};
    }
}

Block Block::onnx(std::string name, bytes graph) {
    return Block{BlockKind::Onnx, std::move(name), {}, std::move(graph)};
}

bool Block::runs_on(std::string_view target, std::string_view arch) const noexcept {
    return !kind.get_target().empty() && kind.get_target() == target &&
           (target_arch.empty() || target_arch == arch);
}

// ── Provenance / Geometry / Rf3Metadata ───────────────────────────────────────────────────────────

void Provenance::encode(Writer& w) const {
    w.string(dataset);
    w.string(software);
    w.string(physics);
    w.string(created);
}

Provenance Provenance::decode(Reader& r) {
    Provenance p;
    p.dataset = r.string("provenance dataset");
    p.software = r.string("provenance software");
    p.physics = r.string("provenance physics");
    p.created = r.string("provenance created");
    return p;
}

void Geometry::encode(Writer& w) const {
    for (const float v : field_dimensions_m) w.f32(v);
    if (!voxelization) {
        w.boolean(false);
        return;
    }
    w.boolean(true);
    for (const auto c : voxelization->voxel_counts) w.u32(c);
    for (const float d : voxelization->voxel_dimensions_m) w.f32(d);
}

Geometry Geometry::decode(Reader& r) {
    Geometry g;
    for (auto& slot : g.field_dimensions_m) slot = r.f32("field dimension");
    if (r.boolean("has voxelization")) {
        Voxelization v;
        for (auto& slot : v.voxel_counts) slot = r.u32("voxel count");
        for (auto& slot : v.voxel_dimensions_m) slot = r.f32("voxel dimension");
        g.voxelization = v;
    }
    return g;
}

void Rf3Metadata::encode(Writer& w) const {
    w.u32(store_version);
    // Each half is length-prefixed, so a reader that meets a producer with extra fields skips what
    // it does not know instead of desynchronising — the same rule the rest of the format follows.
    w.sized([&](Writer& w) {
        w.u64(simulation.primary_particle_count);
        w.string(simulation.geometry);
        w.string(simulation.physics_list);
        for (const float v : simulation.tube.radiation_direction) w.f32(v);
        for (const float v : simulation.tube.radiation_origin) w.f32(v);
        w.f32(simulation.tube.max_energy_ev);
        w.string(simulation.tube.tube_id);
    });
    w.sized([&](Writer& w) {
        w.string(software.name);
        w.string(software.version);
        w.string(software.repository);
        w.string(software.commit);
        w.string(software.doi);
    });
    w.blob(dynamic);
}

Rf3Metadata Rf3Metadata::decode(Reader& r) {
    Rf3Metadata md;
    md.store_version = r.u32("rf3 metadata store version");
    md.simulation = r.sized("rf3 simulation", [](Reader& r) {
        Rf3Simulation s;
        s.primary_particle_count = r.u64("primary particle count");
        s.geometry = r.string("simulation geometry");
        s.physics_list = r.string("simulation physics list");
        for (auto& v : s.tube.radiation_direction) v = r.f32("tube radiation direction");
        for (auto& v : s.tube.radiation_origin) v = r.f32("tube radiation origin");
        s.tube.max_energy_ev = r.f32("tube max energy");
        s.tube.tube_id = r.string("tube id");
        return s;
    });
    md.software = r.sized("rf3 software", [](Reader& r) {
        Rf3Software s;
        s.name = r.string("software name");
        s.version = r.string("software version");
        s.repository = r.string("software repository");
        s.commit = r.string("software commit");
        s.doi = r.string("software doi");
        return s;
    });
    const byte_view dynamic = r.blob("rf3 dynamic metadata");
    md.dynamic.assign(dynamic.begin(), dynamic.end());
    return md;
}

// ── queries ───────────────────────────────────────────────────────────────────────────────────────

std::vector<const TensorDescriptor*> Package::get_inputs() const {
    std::vector<const TensorDescriptor*> out;
    for (const auto& d : io)
        if (d.role == Role::Input) out.push_back(&d);
    return out;
}

std::vector<const TensorDescriptor*> Package::get_outputs() const {
    std::vector<const TensorDescriptor*> out;
    for (const auto& d : io)
        if (d.role == Role::Output) out.push_back(&d);
    return out;
}

const TensorDescriptor* Package::get_tensor(std::string_view name) const noexcept {
    for (const auto& d : io)
        if (d.name == name) return &d;
    return nullptr;
}

const TensorDescriptor* Package::get_tensor_with(Role role, const Semantic& semantic) const noexcept {
    for (const auto& d : io)
        if (d.role == role && d.semantic == semantic) return &d;
    return nullptr;
}

bool Package::is_voxelwise() const noexcept {
    return get_tensor_with(Role::Input, Semantic::Position) != nullptr;
}

ModelKind Package::get_model_kind() const noexcept {
    return is_voxelwise() ? ModelKind::VoxelWise : ModelKind::WholeVolume;
}

std::string_view to_string(ModelKind kind) noexcept {
    switch (kind) {
        case ModelKind::VoxelWise: return "voxelwise";
        case ModelKind::WholeVolume: return "whole_volume";
    }
    return "unknown";
}

std::vector<std::string_view> Package::get_unknown_semantics() const {
    std::vector<std::string_view> out;
    for (const auto& d : io)
        if (!d.semantic.is_known()) out.push_back(d.semantic.get_name());
    return out;
}

std::optional<Composition> Package::get_composition() const {
    for (const auto& b : blocks)
        if (b.kind == BlockKind::Composition) return Composition::from_bytes(byte_view(b.payload));
    return std::nullopt;
}

std::optional<byte_view> Package::get_graph(std::string_view name) const noexcept {
    for (const auto& b : blocks)
        if (b.kind == BlockKind::Onnx && b.name == name) return byte_view(b.payload);
    return std::nullopt;
}

std::vector<std::string_view> Package::get_graph_names() const {
    std::vector<std::string_view> out;
    for (const auto& b : blocks)
        if (b.kind == BlockKind::Onnx) out.push_back(b.name);
    return out;
}

byte_view Package::get_weights(std::string_view stage) const noexcept {
    // The block a stage's parameters live in shares the stage's name, so there is nothing to look up
    // and nothing that can point at the wrong stage's numbers.
    for (const auto& b : blocks)
        if (b.kind == BlockKind::Weights && b.name == stage) return byte_view(b.payload);
    return {};
}

std::vector<std::string_view> Package::get_weights_block_names() const {
    std::vector<std::string_view> out;
    for (const auto& b : blocks)
        if (b.kind == BlockKind::Weights) out.push_back(b.name);
    return out;
}

std::vector<std::string_view> Package::get_executable_block_names() const {
    std::vector<std::string_view> out;
    for (const auto& b : blocks) {
        if (!b.kind.is_executable()) continue;
        if (std::find(out.begin(), out.end(), std::string_view(b.name)) == out.end()) out.push_back(b.name);
    }
    return out;
}

std::vector<std::string> Package::describe_blocks(std::string_view name) const {
    std::vector<std::string> out;
    for (const auto& b : blocks) {
        if (b.name != name || !b.kind.is_executable()) continue;
        out.push_back(std::string(b.kind.get_name()) +
                      (b.target_arch.empty() ? "" : "(" + b.target_arch + ")"));
    }
    return out;
}

bool Package::is_portable() const noexcept {
    const auto composition = get_composition();
    if (!composition) return get_graph(kTrunkGraph).has_value();
    for (const auto& s : composition->stages)
        if (!get_graph(s.name)) return false;
    return true;
}

void Package::require_runnable_on(std::string_view target, std::string_view arch,
                                  std::string_view label) const {
    // With no composition there is one stage — the trunk — which `validate` already proved exists.
    std::vector<std::string_view> stage_names;
    // `get_composition` DECODES the block and returns by value, so the optional must outlive the
    // views taken into its stage names — declaring it inside the `if` condition ends its lifetime
    // at the end of that statement and leaves every view below dangling. Names up to the
    // small-string capacity hide it (the characters sit inside the freed Stage object and are
    // usually still readable); a longer one reads reused heap and the loop below compares garbage,
    // reporting every stage as unrunnable or faulting outright.
    const std::optional<Composition> composition = get_composition();
    if (composition)
        for (const auto& s : composition->stages) stage_names.push_back(s.name);
    else
        stage_names.push_back(kTrunkGraph);

    std::string unsupported;
    for (const auto name : stage_names) {
        // The ONNX form runs under every backend, so a stage that has one is never the problem.
        if (get_graph(name)) continue;
        if (select_block(name, target, arch)) continue;
        const auto carried = describe_blocks(name);
        std::string forms;
        for (const auto& f : carried) forms += (forms.empty() ? "" : ", ") + f;
        unsupported += (unsupported.empty() ? "" : "; ") + ("stage `" + std::string(name) + "` carries " +
                       (forms.empty() ? std::string("no executable form") : forms));
    }
    if (unsupported.empty()) return;
    if (label.empty()) label = target;
    throw Exception::invalid_package(
        "this hardware cannot run the model: " + unsupported + " — none of which runs on " +
        (label.empty() ? std::string("this backend") : std::string(label)) + "/" +
        (arch.empty() ? std::string("unknown architecture") : std::string(arch)));
}

const Block* Package::select_block(std::string_view name, std::string_view target,
                                   std::string_view arch) const noexcept {
    for (const auto& b : blocks)
        if (b.name == name && !b.kind.get_target().empty() && b.kind.get_target() == target && b.target_arch == arch)
            return &b;
    for (const auto& b : blocks)
        if (b.name == name && b.runs_on(target, arch)) return &b;
    return nullptr;
}

void Package::validate() const {
    // SOMETHING must run the trunk. It need not be the ONNX graph: a stage implemented only as a
    // specialised kernel is a package a producer may deliberately ship, and the consequence lands at
    // load, in `require_runnable_on`, as a sentence naming the stage and the device. `is_portable()`
    // is how a producer asks the question this check used to answer by refusing.
    const auto executable = get_executable_block_names();
    if (std::find(executable.begin(), executable.end(), kTrunkGraph) == executable.end())
        throw Exception::invalid_package("no executable block named `" + std::string(kTrunkGraph) +
                                         "`; a package must carry the model in some runnable form");
    if (get_outputs().empty()) throw Exception::invalid_package("declares no outputs");
    for (const auto& d : io) {
        if (d.shape.empty() || std::find(d.shape.begin(), d.shape.end(), 0u) != d.shape.end()) {
            std::string shape;
            for (const auto s : d.shape) shape += (shape.empty() ? "" : ", ") + std::to_string(s);
            throw Exception::invalid_package("tensor `" + d.name + "` has shape [" + shape +
                                         "]; every dimension must be positive");
        }
    }
    std::set<std::string_view> names;
    for (const auto& d : io)
        if (!names.insert(d.name).second)
            throw Exception::invalid_package("tensor name `" + d.name + "` is declared twice");
    if (get_tensor_with(Role::Input, Semantic::SourceDistance) &&
        get_tensor_with(Role::Input, Semantic::SourceOrigin))
        throw Exception::invalid_package("declares both source_distance and source_origin; they are alternatives");
    // The wiring must describe THIS package: a stage naming a graph that is not here, or a
    // connection reading backwards through the recorded order, is a package a consumer could not
    // run. Checked on write as well as on read, so one is never produced.
    std::size_t composition_blocks = 0;
    for (const auto& b : blocks) composition_blocks += b.kind == BlockKind::Composition;
    if (composition_blocks > 1)
        throw Exception::invalid_package("carries " + std::to_string(composition_blocks) +
                                         " composition blocks; a package composes exactly one way");
    if (const auto composition = get_composition())
        composition->validate(get_executable_block_names(), get_weights_block_names());

    // A fixed voxelization only makes sense for a model that emits a whole volume; on a point field
    // it would freeze a resolution the caller is supposed to choose (R-F4).
    if (geometry.voxelization && is_voxelwise())
        throw Exception::invalid_package(
            "records a fixed voxelization but is queried per position; the inference grid is the "
            "caller's choice");
}

// ── the façade ────────────────────────────────────────────────────────────────────────────────────
//
// Three lines each: which version is this, which serializer implements it, call it. No byte walk
// lives here — that is the point of the split (see deploy/version.hpp).

Package Package::read(byte_view bytes) {
    return get_serializer_by(peek_version(bytes)).read(bytes);
}

Metadata Package::read_metadata(byte_view bytes) {
    return get_serializer_by(peek_version(bytes)).read_metadata(bytes);
}

bytes Package::to_bytes() const { return to_bytes(kWriteVersion); }

bytes Package::to_bytes(FormatVersion version) const {
    const Serializer& serializer = get_serializer_by(version);
    if (!serializer.can_write())
        throw Exception::unsupported_version(static_cast<std::uint32_t>(version),
                                             static_cast<std::uint32_t>(kWriteVersion));
    return serializer.write(*this);
}

bytes read_all(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw Exception::io(path.string(), "cannot open for reading");
    bytes out((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (in.bad()) throw Exception::io(path.string(), "read failed");
    return out;
}

Package Package::read_file(const std::filesystem::path& path) { return read(read_all(path)); }

Metadata Package::read_metadata_file(const std::filesystem::path& path) {
    return read_metadata(read_all(path));
}

void Package::write_file(const std::filesystem::path& path) const {
    const bytes data = to_bytes();
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw Exception::io(path.string(), "cannot open for writing");
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!out) throw Exception::io(path.string(), "write failed");
}

}  // namespace RadFiled3D::nn::deploy
