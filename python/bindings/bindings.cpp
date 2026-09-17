// Python bindings (`radfiled3d_nn._rfnn`), pybind11.
//
// The export path is the one that matters first (requirements.md R-P1): `PBRFNet`, `TPBRFNet` and
// their successors in the training framework write their `.rf3m` THROUGH this module, so there is
// exactly one implementation of the byte layout and Python never learns it.
//
// The API mirrors the C++ builder one-for-one and adds no rules of its own. In particular a model
// with an output this library has never heard of is an ordinary `add_output` call — no new enum,
// no new version, nothing to change here.
//
// Every function has a docstring and named arguments: that is what pybind11-stubgen turns into the
// .pyi, so it is the whole of the Python-facing documentation.
#include <RadFiled3D/nn/backends/onnx.hpp>
#include <RadFiled3D/nn/core/field.hpp>
#include <RadFiled3D/nn/core/session.hpp>
#include <RadFiled3D/nn/deploy.hpp>
#include <RadFiled3D/nn/memory/cuda.hpp>
#include <RadFiled3D/nn/memory/memory_ref.hpp>
#include <RadFiled3D/nn/version.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <map>
#include <memory>
#include <optional>

namespace py = pybind11;
using namespace RadFiled3D::nn;
using namespace RadFiled3D::nn::deploy;

namespace {

/// Build a normalizer from the name the training framework uses, plus its parameters. The
/// vocabulary is the C++ one (`normalizer_from_wire`); Python forwards a name and re-decides
/// nothing. A name the library does not know is carried through rather than rejected — the same
/// forward-compatibility rule the format itself follows.
/// The wire name of an element type, as a `DType`. Closed set: a dtype the runtime cannot bind is a
/// hard failure, not something to report and skip (descriptor.hpp).
DType dtype_from_name(const std::string& name) {
    if (name == "f32" || name == "float32") return DType::F32;
    if (name == "f16" || name == "float16") return DType::F16;
    if (name == "i32" || name == "int32") return DType::I32;
    if (name == "u8" || name == "uint8") return DType::U8;
    throw py::value_error("unknown dtype '" + name + "'; expected one of f32, f16, i32, u8");
}

/// A training range from what Python naturally writes for it.
///
/// One argument rather than three mutually exclusive ones, dispatching on the SHAPE of the value:
/// two numbers are an interval, three are a histogram axis, and strings are categories. Each maps to
/// the variant the format already has, so nothing new appears on the wire.
Range range_from_object(const py::object& value) {
    if (value.is_none()) return NoRange{};

    // Strings first: a sequence of them is categorical, and would otherwise be read as numbers.
    if (py::isinstance<py::sequence>(value) && !py::isinstance<py::str>(value)) {
        const auto seq = value.cast<py::sequence>();
        if (py::len(seq) > 0 && py::isinstance<py::str>(seq[0])) {
            Categorical categorical;
            for (const auto& label : seq) categorical.labels.push_back(label.cast<std::string>());
            return categorical;
        }
        switch (py::len(seq)) {
            case 2: return MinMax{seq[0].cast<double>(), seq[1].cast<double>()};
            case 3: return Histogram{seq[0].cast<double>(), seq[1].cast<double>(), seq[2].cast<double>()};
            default: break;
        }
    }
    throw py::value_error(
        "range must be (min, max), (min, max, bin_width) for a histogram axis, a sequence of "
        "category labels, or None");
}

/// What kind of range this is, by name, even where there is no Python shape for its value.
///
/// Rule 5: a kind a build does not recognise is preserved verbatim rather than dropped, and a
/// consumer should be able to SAY that a package declares one — reporting `None` for both the value
/// and the kind would make "no range" and "a range I cannot render" look identical.
std::string_view range_kind_name(const Range& range) {
    return std::visit(
        [](const auto& r) -> std::string_view {
            using T = std::decay_t<decltype(r)>;
            if constexpr (std::is_same_v<T, NoRange>) return "none";
            else if constexpr (std::is_same_v<T, MinMax>) return "min_max";
            else if constexpr (std::is_same_v<T, Histogram>) return "histogram";
            else if constexpr (std::is_same_v<T, Categorical>) return "categorical";
            else if constexpr (std::is_same_v<T, RangeMap>) return "map";
            else return "unknown";
        },
        range);
}

/// The inverse, for reporting what a package declares.
py::object range_to_object(const Range& range) {
    return std::visit(
        [](const auto& r) -> py::object {
            using T = std::decay_t<decltype(r)>;
            if constexpr (std::is_same_v<T, MinMax>) return py::make_tuple(r.min, r.max);
            else if constexpr (std::is_same_v<T, Histogram>)
                return py::make_tuple(r.min, r.max, r.bin_width);
            else if constexpr (std::is_same_v<T, Categorical>) return py::cast(r.labels);
            else
                // NoRange, a named group, or a kind written by a newer producer. The bytes are
                // preserved either way (rule 5); there is simply no Python shape for them yet.
                return py::none();
        },
        range);
}

TensorDescriptor make_descriptor(Role role, const std::string& name, const std::string& semantic,
                                 std::vector<std::uint32_t> shape, const std::string& unit,
                                 const py::object& range, const std::string& normalizer,
                                 std::optional<std::vector<double>> normalizer_params, bool optional,
                                 const std::string& dtype) {
    TensorDescriptor d;
    d.name = name;
    d.role = role;
    d.semantic = Semantic::from_name(semantic);
    d.shape = std::move(shape);
    d.unit = unit;
    d.dtype = dtype_from_name(dtype);
    d.range = range_from_object(range);
    d.normalizer = normalizer_from_wire(normalizer, normalizer_params.value_or(std::vector<double>{}));
    d.optional = optional;
    return d;
}

py::dict tensor_dict(const TensorDescriptor& d) {
    py::dict entry;
    entry["name"] = d.name;
    entry["role"] = std::string(to_string(d.role));
    entry["semantic"] = std::string(d.semantic.get_name());
    entry["known_semantic"] = d.semantic.is_known();
    entry["shape"] = d.shape;
    entry["dtype"] = std::string(to_string(d.dtype));
    entry["unit"] = d.unit;
    entry["normalizer"] = std::string(get_name(d.normalizer));
    entry["normalizer_params"] = get_params(d.normalizer);
    entry["range"] = range_to_object(d.range);
    entry["range_kind"] = std::string(range_kind_name(d.range));
    entry["optional"] = d.optional;
    return entry;
}

// Static: pybind11 keeps the docstring pointer, so it must outlive the module initialisation.
// ── binding caller-owned buffers ─────────────────────────────────────────────────────────────────
//
// A buffer reaches inference BY POINTER and is never copied, so what these do is find the pointer a
// Python object already has and refuse anything laid out differently from how the runtime will read
// it. Four protocols, in the order that gets it right:
//
//   1. __cuda_array_interface__ — a torch CUDA tensor, a CuPy array. Device memory.
//   2. __array_interface__      — a NumPy array. Host memory.
//   3. torch's own accessors    — duck-typed (`data_ptr`/`is_cuda`/`numel`), so a CPU torch tensor
//                                 works without this module importing or depending on torch.
//   4. the buffer protocol      — anything else contiguous.
//
// Non-contiguous input is REFUSED, never quietly copied: a copy is a different buffer, and the
// caller asked for theirs to be written.

/// A buffer the session can bind, plus the Python object that owns it. Holding `owner` IS the
/// lifetime contract — a `MemoryRef` points at memory it does not own, so something must keep the
/// array alive for as long as the binding lasts, and this is it.
struct PyMemory {
    std::shared_ptr<memory::MemoryRef> ref;
    py::object owner;
};

/// `(pointer, bytes)` from an array-interface dict, or nullopt when `obj` has no such attribute.
std::optional<std::pair<std::uint64_t, std::uint64_t>> from_array_interface(const py::object& obj,
                                                                           const char* attribute) {
    if (!py::hasattr(obj, attribute)) return std::nullopt;
    const py::dict interface = obj.attr(attribute).cast<py::dict>();
    // A null `strides` means C-contiguous. Anything else describes a layout the runtime would read
    // as though it were dense.
    if (interface.contains("strides") && !interface["strides"].is_none())
        throw py::value_error(
            "the buffer is not contiguous; inference binds memory by pointer and will not copy. "
            "Pass a contiguous array (numpy.ascontiguousarray / Tensor.contiguous()).");
    const auto typestr = interface["typestr"].cast<std::string>();
    // float32 only: every declared tensor is F32, and silently reinterpreting f2 or f8 would be a
    // wrong result rather than a conversion.
    if (typestr != "<f4" && typestr != "=f4" && typestr != "float32")
        throw py::value_error("expected a float32 buffer, got dtype '" + typestr + "'");
    const auto data = interface["data"].cast<py::tuple>();
    const auto pointer = data[0].cast<std::uint64_t>();
    std::uint64_t elements = 1;
    for (const auto dim : interface["shape"].cast<py::tuple>()) elements *= dim.cast<std::uint64_t>();
    return std::pair<std::uint64_t, std::uint64_t>{pointer, elements * sizeof(float)};
}

PyMemory to_memory(const py::object& obj) {
    if (py::isinstance<PyMemory>(obj)) return obj.cast<PyMemory>();

    // Device memory FIRST: a torch CUDA tensor can answer both interfaces, and reading the host one
    // would hand the runtime a device pointer to dereference on the CPU.
    if (const auto cuda = from_array_interface(obj, "__cuda_array_interface__"))
        // `__cuda_array_interface__` carries an address and a length and NO ordinal, so the driver is
        // asked which card the pointer is on. Without that a tensor on `cuda:1` binds to a device-0
        // session and the session has nothing to refuse it with.
        return {std::make_shared<memory::cuda::MemoryRef>(cuda->first, cuda->second,
                                                          memory::cuda::device_of(cuda->first)),
                obj};
    if (const auto host = from_array_interface(obj, "__array_interface__"))
        return {std::make_shared<memory::host::MemoryRef>(
                    reinterpret_cast<void*>(static_cast<std::uintptr_t>(host->first)), host->second),
                obj};

    // torch, duck-typed. This module must not depend on torch — a consumer that only reads packages
    // should never pay for it (R-P1) — so a tensor is recognised by what it can do, not by its type.
    if (py::hasattr(obj, "data_ptr") && py::hasattr(obj, "is_cuda") && py::hasattr(obj, "numel") &&
        py::hasattr(obj, "element_size")) {
        if (py::hasattr(obj, "is_contiguous") && !obj.attr("is_contiguous")().cast<bool>())
            throw py::value_error("the tensor is not contiguous; call .contiguous() first");
        const auto pointer = obj.attr("data_ptr")().cast<std::uint64_t>();
        const auto bytes = obj.attr("numel")().cast<std::uint64_t>() *
                           obj.attr("element_size")().cast<std::uint64_t>();
        if (obj.attr("is_cuda").cast<bool>())
            return {std::make_shared<memory::cuda::MemoryRef>(pointer, bytes,
                                                              memory::cuda::device_of(pointer)),
                    obj};
        return {std::make_shared<memory::host::MemoryRef>(
                    reinterpret_cast<void*>(static_cast<std::uintptr_t>(pointer)), bytes), obj};
    }

    const auto buffer = py::reinterpret_borrow<py::buffer>(obj);
    const py::buffer_info info = buffer.request();
    if (info.format != py::format_descriptor<float>::format())
        throw py::value_error("expected a float32 buffer");
    return {std::make_shared<memory::host::MemoryRef>(
                info.ptr, static_cast<std::uint64_t>(info.size) * sizeof(float)),
            obj};
}

/// A loaded model. Owns the bound Python objects, so what inference writes into cannot be collected
/// while it runs.
class PySession {
public:
    PySession(Package package, const std::string& backend, int device)
        : session_(onnx::load(std::move(package), backend_from_name(backend), device)) {}

    void set_voxel_grid(std::array<std::uint32_t, 3> counts) {
        session_->set_voxel_grid(counts);
        // The session dropped its bindings; drop our references with them rather than pinning
        // arrays nothing is bound to any more.
        bound_.clear();
    }

    void bind_input(const std::string& name, const py::object& obj) { bind(name, obj, true); }
    void bind_output(const std::string& name, const py::object& obj) { bind(name, obj, false); }

    void infer() {
        // Long, and touches no Python object — so the interpreter is released for its duration.
        // Without this one inference blocks every other thread in the process.
        py::gil_scoped_release unlocked;
        session_->infer();
    }

    std::string get_backend() const { return std::string(to_string(session_->get_backend())); }
    int get_device() const { return session_->get_device(); }

private:
    void bind(const std::string& name, const py::object& obj, bool is_input) {
        PyMemory memory = to_memory(obj);
        if (is_input)
            session_->bind_input(name, memory.ref);
        else
            session_->bind_output(name, memory.ref);
        bound_[(is_input ? "in:" : "out:") + name] = memory.owner;
    }

    std::unique_ptr<InferenceSession> session_;
    std::map<std::string, py::object> bound_;
};

// ── the GPU field ────────────────────────────────────────────────────────────────────────────────
//
// `RadFiled3D::GPUCartesianRadiationField` is a real `RadiationField<BufferT>` subclass, so it can be
// handed to anything in RadFiled3D unchanged. This module registers IT, and deliberately does NOT
// register `CartesianRadiationField`: RadFiled3D's own bindings already do, and pybind11 refuses a
// second registration of the same C++ type with "already registered".
//
// That is also what makes `to_host_field` work across the two modules. pybind11 keeps its type
// registry in per-process shared internals keyed by its version and ABI, so a host field converts
// to RADFILED3D'S OWN Python class — no wrapper, no copy, no duplicated API. Two things have to
// hold, and `python/CMakeLists.txt` is what makes the second one true:
//   * the type must be registered, which happens when `RadFiled3D.RadFiled3D` (the native module,
//     not the package `__init__`) is imported — `to_host_field` imports it itself;
//   * both modules must have been built against the SAME pybind11, which is why this build takes
//     its pybind11 pin from RadFiled3D rather than choosing one.
// Where either fails the conversion says so rather than inventing a type.

/// Zero-copy view of a float layer. The array borrows the field's buffer, and `base` keeps the field
/// alive for exactly as long as any view of it — filling the array fills the voxels.
py::array_t<float> layer_view(const std::shared_ptr<RadFiled3D::GPUCartesianRadiationField>& field,
                              const std::string& channel, const std::string& layer) {
    // RadFiled3D THROWS for an absent channel rather than returning null, so ask first: a missing
    // channel and a missing layer are one kind of mistake and must raise one kind of exception.
    if (!field->has_channel(channel)) throw py::key_error("no channel named '" + channel + "'");
    auto buffer = field->get_channel(channel);
    if (!buffer->has_layer(layer)) throw py::key_error("no layer named '" + layer + "'");
    float* data = buffer->get_layer<float>(layer);
    const auto voxels = static_cast<py::ssize_t>(buffer->get_voxel_count());
    return py::array_t<float>({voxels}, {sizeof(float)}, data, py::cast(field));
}

const std::string kTensorArgs =
    "\n\nArgs:\n"
    "    name: the graph tensor name; the runtime binds by it.\n"
    "    semantic: what the numbers mean — 'position', 'beam_direction', 'flux', ... or any new name.\n"
    "    shape: per-query shape; resolution lives here (e.g. [bins] for a spectrum).\n"
    "    unit: physical unit of the metric value: 'm', 'deg', 'rad', 'eV', or ''.\n"
    "    range: the training range — what the model generalises over, so a query outside it can be\n"
    "        reported as out of distribution. Its SHAPE says which kind:\n"
    "            (min, max)             an interval\n"
    "            (min, max, bin_width)  a histogram axis, e.g. a tube spectrum in eV\n"
    "            ['a', 'b', ...]        category labels\n"
    "            None                   not recorded\n"
    "    normalizer: 'identity', 'linear0_1', 'linear-1_1', 'log_scale', 'asinh', or a custom name.\n"
    "    normalizer_params: the normalizer's parameters, e.g. [min, max] or [epsilon, scale].\n"
    "    optional: whether the runtime may leave the tensor unbound.\n"
    "    dtype: element type — 'f32' (default), 'f16', 'i32' or 'u8'. Unlike a semantic this set is\n"
    "        closed: a dtype the runtime cannot bind is a hard failure, not something to skip.";
const std::string kAddInputDoc = "Declare an input tensor." + kTensorArgs;
const std::string kAddOutputDoc =
    "Declare an output tensor. A quantity this library has never heard of is an ordinary call." + kTensorArgs;

}  // namespace

PYBIND11_MODULE(_rfnn, m) {
    m.doc() = "RF3M — read and write the deployment container for neural radiation fields.";

    // IOError for a file problem, ValueError for everything about the package itself.
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const RadFiled3D::nn::Exception& err) {
            if (err.get_kind() == RadFiled3D::nn::ErrorKind::Io)
                PyErr_SetString(PyExc_IOError, err.what());
            else
                PyErr_SetString(PyExc_ValueError, err.what());
        }
    });

    m.def("version", [] { return std::string(RadFiled3D::nn::kVersion); }, "The library version.");

    // ── submodules, mirroring the C++ namespaces ─────────────────────────────────────────────────
    //
    // `RadFiled3D::nn::deploy` -> `RadFiled3D.nn.deploy`, `::memory` -> `.memory`, and `core/` — the
    // sessions and the fields — reaches Python as `.inference`, which is what it does. A reader who
    // knows one side then knows the other, which is the whole point of matching the layout.
    //
    // A `GPUCartesianRadiationField` stays on the PLAIN module because in C++ it is declared in
    // namespace `RadFiled3D` itself, not in a sub-namespace: it IS a RadFiled3D field (rule 7b), and
    // putting it under `.inference` here would claim otherwise.
    //
    // These are attributes of the extension, not importable modules. The thin Python shims in
    // `RadFiled3D/nn/` re-export them, which is what makes `import RadFiled3D.nn.deploy` work —
    // `def_submodule` alone creates no `sys.modules` entry.
    auto deploy = m.def_submodule("deploy", "The .rf3m container: describe a model, write it, read it back.");
    auto inference = m.def_submodule("inference", "Run a package on a device and bind its tensors.");
    auto memory = m.def_submodule("memory", "Buffers inference reads and writes, bound by pointer.");

    py::class_<PackageBuilder>(deploy, "PackageBuilder", "Assemble a deployment package and write it as .rf3m.")
        .def(py::init([](const std::string& dataset, const std::string& software, const std::string& physics,
                         const std::string& created) {
                 auto b = std::make_unique<PackageBuilder>();
                 b->provenance(dataset, software, physics).created(created);
                 return b;
             }),
             py::arg("dataset") = "", py::arg("software") = "", py::arg("physics") = "", py::arg("created") = "",
             "Start a package. `created` is an ISO-8601 UTC timestamp, supplied rather than read from the "
             "clock so that building the same model twice can produce the same bytes.")
        .def(
            "set_field_dimensions",
            [](PackageBuilder& b, std::array<float, 3> dims) { b.field_dimensions_m(dims); }, py::arg("dimensions_m"),
            "The metric box that normalised positions span, in metres.")
        .def(
            "set_voxelization",
            [](PackageBuilder& b, std::array<std::uint32_t, 3> counts, std::array<float, 3> dims) {
                b.voxelization(counts, dims);
            },
            py::arg("voxel_counts"), py::arg("voxel_dimensions_m"),
            "Record a fixed voxel grid. Only valid for a whole-volume model; `write` rejects it on a model "
            "queried per position.")
        .def(
            "add_input",
            [](PackageBuilder& b, const std::string& name, const std::string& semantic, std::vector<std::uint32_t> shape,
               const std::string& unit, const py::object& range, const std::string& normalizer,
               std::optional<std::vector<double>> normalizer_params, bool optional,
               const std::string& dtype) {
                b.descriptor(make_descriptor(Role::Input, name, semantic, std::move(shape), unit, range, normalizer,
                                             std::move(normalizer_params), optional, dtype));
            },
            py::arg("name"), py::arg("semantic"), py::arg("shape"), py::arg("unit") = "", py::arg("range") = py::none(),
            py::arg("normalizer") = "identity", py::arg("normalizer_params") = py::none(),
            py::arg("optional") = false, py::arg("dtype") = "f32",
            kAddInputDoc.c_str())
        .def(
            "add_output",
            [](PackageBuilder& b, const std::string& name, const std::string& semantic, std::vector<std::uint32_t> shape,
               const std::string& unit, const py::object& range, const std::string& normalizer,
               std::optional<std::vector<double>> normalizer_params, bool optional,
               const std::string& dtype) {
                b.descriptor(make_descriptor(Role::Output, name, semantic, std::move(shape), unit, range, normalizer,
                                             std::move(normalizer_params), optional, dtype));
            },
            py::arg("name"), py::arg("semantic"), py::arg("shape"), py::arg("unit") = "", py::arg("range") = py::none(),
            py::arg("normalizer") = "identity", py::arg("normalizer_params") = py::none(),
            py::arg("optional") = false, py::arg("dtype") = "f32",
            kAddOutputDoc.c_str())
        .def(
            "add_graph",
            [](PackageBuilder& b, const std::string& name, py::bytes onnx) {
                const std::string_view s = onnx;
                b.graph(name, bytes(s.begin(), s.end()));
            },
            py::arg("name"), py::arg("onnx"), "Attach a named ONNX graph. `trunk` is mandatory.")
        .def(
            "add_block",
            [](PackageBuilder& b, const std::string& kind, const std::string& name, py::bytes payload,
               const std::string& target_arch) {
                const std::string_view s = payload;
                b.block(BlockKind::from_name(kind), name, target_arch, bytes(s.begin(), s.end()));
            },
            py::arg("kind"), py::arg("name"), py::arg("payload"), py::arg("target_arch") = "",
            "Attach a specialised executable form of a stage — `kind` is 'cuda_ptx', 'cuda_cubin', 'hsaco', "
            "'spirv', 'opencl', or any name a newer runtime understands. `target_arch` is e.g. 'sm_90', or '' "
            "for a form that runs on any architecture of its kind.\n\n"
            "Share the `name` with the stage it implements. A stage needs only ONE form: shipping a "
            "kernel and no ONNX graph is legitimate, and the consequence is that loading on hardware "
            "none of the forms runs on fails with a message naming that stage.")
        .def(
            "add_buffer",
            [](PackageBuilder& b, const std::string& name, std::uint64_t elements) {
                b.buffer(name, elements);
                return &b;
            },
            py::arg("name"), py::arg("elements"), py::return_value_policy::reference,
            "Declare a named buffer that carries a value from one stage to later ones.\n\n"
            "Args:\n"
            "    name: what stages refer to it by.\n"
            "    elements: how many floats ONE ROW holds. How many rows it holds is derived when\n"
            "        the stages are wired, from the shapes the producing and consuming graphs\n"
            "        declare, so a value computed once and read at every voxel is repeated and one\n"
            "        read at the same extent is carried as-is.\n\n"
            "The buffer is allocated once, when a grid is chosen, and stays valid for as long as the "
            "model is loaded at that resolution.")
        .def(
            "add_stage",
            [](PackageBuilder& b, const std::string& name, const std::string& invocation,
               const std::map<std::string, std::string>& reads,
               const std::map<std::string, std::string>& writes) {
                StageBuilder stage = b.stage(name, invocation_from_name(invocation));
                for (const auto& [tensor, buffer] : reads) stage.reads(tensor, buffer);
                for (const auto& [tensor, buffer] : writes) stage.writes(tensor, buffer);
                return &b;
            },
            py::arg("name"), py::arg("invocation") = "per_query",
            py::arg("reads") = std::map<std::string, std::string>{},
            py::arg("writes") = std::map<std::string, std::string>{},
            py::return_value_policy::reference,
            "Record that a stage runs, how often, and which buffers it uses. Call once per stage, "
            "IN EXECUTION ORDER; the `trunk` runs last and per query.\n\n"
            "Args:\n"
            "    name: the stage's identity AND the name its blocks were added under. A stage may\n"
            "        be implemented by an ONNX graph, by a specialised kernel, or by several — the\n"
            "        one that runs is chosen against the device at load.\n"
            "    invocation: 'once' for a stage evaluated once per field — a beam encoder, an\n"
            "        encoding config — whose result is reused for every voxel, or 'per_query' for\n"
            "        one evaluated at every voxel. This is NOT derivable from the graph: a beam\n"
            "        encoder declares a batch axis on every input and still runs once, because a\n"
            "        beam is constant over the volume.\n"
            "    reads: {tensor: buffer} fed from buffers EARLIER stages wrote. A stage cannot read\n"
            "        a buffer written later or by itself — the value would not exist yet.\n"
            "    writes: {tensor: buffer} this stage fills for later stages. A tensor listed in\n"
            "        neither is the caller's to bind.\n\n"
            "Tensor names here are GRAPH names, which are frequently not declared tensors at all — a "
            "caller never supplies a latent and never sees one. Recording the wiring is what lets a "
            "consumer run a factored model from the package alone, including when the names do not "
            "match: the packages already on disk call the encoder's output `linear_5` while the "
            "trunk's input is `latent`.\n\n"
            "A package with no stages composes the way every package always has — a single trunk, "
            "driven directly — so a one-graph model needs none of this.")
        .def(
            "add_stage_weights",
            [](PackageBuilder& b, const std::string& stage, py::bytes payload, std::uint64_t elements,
               const std::string& dtype, const std::string& onnx_external_file) {
                const std::string_view s = payload;
                b.edit_stage(stage).weights(bytes(s.begin(), s.end()), elements, dtype_from_name(dtype),
                                            onnx_external_file);
                return &b;
            },
            py::arg("stage"), py::arg("payload"), py::arg("elements"), py::arg("dtype") = "f32",
            py::arg("onnx_external_file") = "", py::return_value_policy::reference,
            "Attach a stage's parameters — the ONE buffer every implementation of that stage reads.\n\n"
            "A stage may be evaluated by a PTX kernel on NVIDIA, an HSACO object on AMD and an ONNX "
            "graph anywhere. Those are three ways to compute the same function, so they read one set "
            "of numbers: a copy per implementation would be three things that can drift apart.\n\n"
            "Args:\n"
            "    stage: the stage these belong to; it must already have been added. The parameters\n"
            "        are stored under a `weights` block sharing its name, and no other stage reads\n"
            "        them.\n"
            "    payload: the raw bytes, in whatever layout the implementations agree on.\n"
            "    elements: how many values they are, so a consumer knows the layout without parsing\n"
            "        the payload.\n"
            "    dtype: the element type they all agree on — 'f32', 'f16', 'i32', 'u8'. ONE type for\n"
            "        the whole buffer: a kernel reading it as f16 while the graph reads f32 is the\n"
            "        drift this design exists to prevent.\n"
            "    onnx_external_file: the external-data file name an ONNX implementation refers to\n"
            "        them by. A graph exported with external data records a file name instead of\n"
            "        embedding its initializers; give that name here and the runtime satisfies it\n"
            "        from the package in memory, so nothing has to sit on disk beside the .rf3m.\n"
            "        Leave empty for a graph that embeds its own initializers.\n\n"
            "A stage with no parameters needs none of this: it has an EMPTY buffer, and the runtime "
            "hands back an empty view rather than nothing, so an implementation reads its weights "
            "the same way whether or not it has any.")
        .def(
            "add_stage_launch",
            [](PackageBuilder& b, const std::string& stage, const std::string& kind,
               const std::string& entry_point, const std::string& target_arch,
               std::array<std::uint32_t, 3> block_size, std::array<std::uint32_t, 3> grid_size,
               std::uint32_t shared_memory_bytes) {
                KernelLaunch launch;
                launch.entry_point = entry_point;
                launch.block_size = block_size;
                launch.grid_size = grid_size;
                launch.shared_memory_bytes = shared_memory_bytes;
                b.edit_stage(stage).launch(BlockKind::from_name(kind), target_arch, std::move(launch));
                return &b;
            },
            py::arg("stage"), py::arg("kind"), py::arg("entry_point"), py::arg("target_arch") = "",
            py::arg("block_size") = std::array<std::uint32_t, 3>{256, 1, 1},
            py::arg("grid_size") = std::array<std::uint32_t, 3>{0, 0, 0},
            py::arg("shared_memory_bytes") = 0u, py::return_value_policy::reference,
            "Say how to launch one compiled form of a stage.\n\n"
            "Needed only for a stage implemented by a kernel block — an ONNX graph carries its own "
            "calling convention and declaring one beside it is refused. A stage with several "
            "compiled forms declares one launch each, because a CUDA kernel and an HSACO one need "
            "not share an entry symbol or a block size.\n\n"
            "Args:\n"
            "    stage: the stage this describes; it must already have been added.\n"
            "    kind: which compiled form — 'cuda_ptx', 'cuda_cubin', 'hsaco', 'spirv', 'opencl'.\n"
            "    entry_point: the symbol to launch in the loaded module.\n"
            "    target_arch: e.g. 'sm_90', or '' for the form that runs on any architecture.\n"
            "    grid_size: a zero component is derived from the row count the stage runs at.")
        .def(
            "add_metric", [](PackageBuilder& b, const std::string& name, double value) { b.metric(name, value); },
            py::arg("name"), py::arg("value"), "Record a test metric measured after training.")
        .def(
            "get_declared_tensors",
            [](const PackageBuilder& b) {
                py::list out;
                for (const auto& d : b.get_declared_tensors()) out.append(tensor_dict(d));
                return out;
            },
            "The tensors declared so far, as dicts. Lets a producer check its exported graphs against what the "
            "package promises (see radfiled3d_nn.torch_export.verify_declarations). Names only, never payloads.")
        .def(
            "to_bytes",
            [](const PackageBuilder& b) {
                const bytes data = b.build().to_bytes();
                return py::bytes(reinterpret_cast<const char*>(data.data()), data.size());
            },
            "Serialise to bytes, validating first.")
        .def(
            "write", [](const PackageBuilder& b, const std::string& path) { b.build().write_file(path); }, py::arg("path"),
            "Write the package. A package that would fail to load is never written.")
        .def("__repr__", [](const PackageBuilder& b) {
            std::size_t inputs = 0;
            for (const auto& d : b.get_declared_tensors()) inputs += d.role == Role::Input;
            return "PackageBuilder(inputs=" + std::to_string(inputs) +
                   ", outputs=" + std::to_string(b.get_declared_tensors().size() - inputs) +
                   ", graphs=" + std::to_string(b.get_graph_count()) + ")";
        });

    py::class_<PyMemory>(memory, "Memory",
                         "A buffer inference can write into, referenced BY POINTER and never copied.\n\n"
                         "Wraps a NumPy array, a torch tensor (CPU or CUDA), or anything else exposing "
                         "`__array_interface__`, `__cuda_array_interface__` or the buffer protocol. The "
                         "wrapped object is kept alive for as long as the Memory is, but the memory is "
                         "still the caller's: this never allocates and never frees.")
        .def(py::init([](const py::object& buffer) { return to_memory(buffer); }), py::arg("buffer"),
             "Wrap a buffer. Raises ValueError if it is not contiguous float32 — a non-contiguous "
             "array would have to be copied, and a copy is not the buffer the caller asked to have "
             "written.")
        .def_property_readonly("size_bytes",
                               [](const PyMemory& m) { return m.ref->get_size_bytes(); })
        .def_property_readonly(
            "domain", [](const PyMemory& m) { return std::string(memory::to_string(m.ref->get_domain())); },
            "Where the memory lives: 'host', 'cuda', ... A torch CUDA tensor binds as device memory, "
            "so nothing crosses the bus at inference.")
        .def("__repr__", [](const PyMemory& m) {
            return "Memory(domain='" + std::string(memory::to_string(m.ref->get_domain())) +
                   "', size_bytes=" + std::to_string(m.ref->get_size_bytes()) + ")";
        });

    py::class_<RadFiled3D::GPUCartesianRadiationField,
               std::shared_ptr<RadFiled3D::GPUCartesianRadiationField>>(
        m, "GPUCartesianRadiationField",
        "A Cartesian radiation field whose layers can carry a GPU mirror.\n\n"
        "It IS a RadFiled3D field — the C++ type derives from RadiationField<BufferT> — so anything "
        "downstream of RadFiled3D works on a generated field exactly as on a simulated one. The host "
        "voxels stay the source of truth; a device mirror is what inference writes into.")
        .def(py::init([](std::array<std::uint32_t, 3> voxel_counts,
                         std::array<float, 3> field_dimensions_m) {
                 return allocate_gpu_field(CartesianFieldGeometry::make(voxel_counts, field_dimensions_m));
             }),
             py::arg("voxel_counts"), py::arg("field_dimensions_m"),
             "Allocate a field. The box is METRIC and the voxel size follows from the resolution, "
             "never the other way round; a grid that does not tile the box is refused.")
        .def_property_readonly("voxel_counts",
                               [](const RadFiled3D::GPUCartesianRadiationField& f) {
                                   const auto c = f.get_voxel_counts();
                                   return std::array<std::uint32_t, 3>{c.x, c.y, c.z};
                               })
        .def_property_readonly("field_dimensions_m",
                               [](const RadFiled3D::GPUCartesianRadiationField& f) {
                                   const auto d = f.get_field_dimensions();
                                   return std::array<float, 3>{d.x, d.y, d.z};
                               })
        .def_property_readonly("voxel_dimensions_m",
                               [](const RadFiled3D::GPUCartesianRadiationField& f) {
                                   const auto d = f.get_voxel_dimensions();
                                   return std::array<float, 3>{d.x, d.y, d.z};
                               })
        .def_property_readonly("voxel_count", &RadFiled3D::GPUCartesianRadiationField::get_voxel_count)
        .def(
            "add_channel",
            [](RadFiled3D::GPUCartesianRadiationField& f, const std::string& name) { f.add_channel(name); },
            py::arg("name"), "Add a named channel, or keep the existing one.")
        .def(
            "add_layer",
            [](RadFiled3D::GPUCartesianRadiationField& f, const std::string& channel,
               const std::string& layer, const std::string& unit, float initial) {
                f.add_channel(channel);
                f.get_channel(channel)->add_layer<float>(layer, initial, unit);
            },
            py::arg("channel"), py::arg("layer"), py::arg("unit") = "", py::arg("initial") = 0.f,
            "Add a float32 layer, one element per voxel.")
        .def("layer", &layer_view, py::arg("channel"), py::arg("layer"),
             "The layer's voxels as a NumPy view — no copy, so writing into it writes the field.")
        .def(
            "set_device_memory",
            [](RadFiled3D::GPUCartesianRadiationField& f, const std::string& channel,
               const std::string& layer, const PyMemory& memory) {
                f.set_device_memory(channel, layer, memory.ref);
            },
            py::arg("channel"), py::arg("layer"), py::arg("memory"),
            "Attach the GPU mirror of a layer — the buffer inference writes into. Opaque: only the "
            "backend that produced it knows what it means.")
        .def(
            "get_device_memory",
            [](const RadFiled3D::GPUCartesianRadiationField& f, const std::string& channel,
               const std::string& layer) -> py::object {
                auto ref = f.get_device_memory(channel, layer);
                if (!ref) return py::none();
                return py::cast(PyMemory{std::move(ref), py::none()});
            },
            py::arg("channel"), py::arg("layer"), "The layer's GPU mirror, or None.")
        .def(
            "to_host_field",
            [](const RadFiled3D::GPUCartesianRadiationField& f) -> py::object {
                auto host = f.to_host_field();
                // The type is registered by RadFiled3D's NATIVE module, not by its package
                // `__init__`, so importing `RadFiled3D` alone leaves the registry without it.
                // Import the submodule here rather than making the caller know that; if it is
                // already imported this is a dict lookup.
                std::string import_failure;
                try {
                    py::module_::import("RadFiled3D.RadFiled3D");
                } catch (const py::error_already_set& err) {
                    // Keep WHY. "not installed" and "its .so could not resolve a dependency" lead to
                    // completely different fixes, and the cast below cannot tell them apart.
                    import_failure = err.what();
                }
                try {
                    // No wrapper: this converts to RadFiled3D's OWN CartesianRadiationField, because
                    // their module registered the type and pybind11 shares its registry per process.
                    return py::cast(host);
                } catch (const py::cast_error&) {
                    std::string message =
                        "to_host_field() returns a RadFiled3D.RadFiled3D.CartesianRadiationField, "
                        "which only converts when the RadFiled3D Python package is installed and was "
                        "built against the same pybind11 as this module. Use store_rf3() instead to "
                        "write the field without needing that type at all.";
                    if (!import_failure.empty())
                        message += "\n\nImporting RadFiled3D.RadFiled3D failed with: " + import_failure;
                    throw py::import_error(message);
                }
            },
            "Copy every channel and layer into a host RadFiled3D.CartesianRadiationField.\n\n"
            "GPU mirrors are NOT transferred: the host voxels are the source of truth, so a "
            "device-only result must be synced back through its backend first.")
        .def(
            "store_rf3",
            [](const RadFiled3D::GPUCartesianRadiationField& f, const std::string& path,
               const std::string& software, const std::string& version, const std::string& repository,
               const std::string& commit) {
                store_host_field(f.to_host_field(), path, {software, version, repository, commit});
            },
            py::arg("path"), py::arg("software") = "radfiled3d-nn", py::arg("version") = "",
            py::arg("repository") = "", py::arg("commit") = "",
            "Write the field as .rf3 through RadFiled3D's own FieldStore.\n\n"
            "Self-contained: unlike to_host_field() this needs no RadFiled3D Python module, so it "
            "works in a process that only has this one.")
        .def("__repr__", [](const RadFiled3D::GPUCartesianRadiationField& f) {
            const auto c = f.get_voxel_counts();
            return "GPUCartesianRadiationField(voxel_counts=[" + std::to_string(c.x) + ", " +
                   std::to_string(c.y) + ", " + std::to_string(c.z) + "])";
        });

    py::class_<PySession>(inference, "Session",
                          "A model loaded onto a device and ready to be driven.\n\n"
                          "The protocol, in order:  set_voxel_grid -> bind_input/bind_output -> infer.\n"
                          "The session allocates no result and copies nothing: it writes into the "
                          "buffers that were bound. Bindings are read again on every run, so editing an "
                          "input in place needs no rebinding — but changing the grid clears them, "
                          "because their shapes depended on it.")
        .def("set_voxel_grid", &PySession::set_voxel_grid, py::arg("voxel_counts"),
             "Choose the grid the next inference fills. Clears every binding.")
        .def("bind_input", &PySession::bind_input, py::arg("name"), py::arg("buffer"),
             "Register an input. It holds METRIC values — metres, radians, electronvolts — and the "
             "package's normalizer is applied by the session, so a caller never learns how the model "
             "was normalised.")
        .def("bind_output", &PySession::bind_output, py::arg("name"), py::arg("buffer"),
             "Register where an output is written. Nothing is allocated for you.")
        .def("infer", &PySession::infer,
             "Run. Releases the GIL for the duration. A missing binding raises, and the message names "
             "every one that is missing rather than the first.")
        .def_property_readonly("backend", &PySession::get_backend)
        .def_property_readonly("device", &PySession::get_device,
                               "The device ordinal everything in this session runs on — the execution "
                               "provider, the buffers between stages, and any kernel stage's launches. "
                               "Always a real index: 'automatic' is resolved when the session is built. "
                               "Memory from another device is refused rather than faulted on.");

    inference.def(
        "load_rf3m",
        [](const std::string& path, const std::string& backend, int device) {
            return std::make_unique<PySession>(Package::read_file(path), backend, device);
        },
        py::arg("path"), py::arg("backend") = "cpu", py::arg("device") = -1,
        "Load a package and build a session on `backend` — 'cpu', 'cuda', 'tensorrt', 'rocm' or "
        "'directml'.\n\n"
        "`device` is the ordinal EVERYTHING runs on: the execution provider, the buffers between "
        "stages, and any kernel stage's module and launches. -1 means the default device, and the "
        "resolved index is readable afterwards as `session.device`. Binding GPU memory from a "
        "different device raises rather than faulting — a torch tensor on 'cuda:1' is refused by a "
        "device-0 session, naming both — so load the session on the device your tensors are already "
        "on. An ordinal the backend does not have raises, naming how many it has.\n\n"
        "Raises ValueError naming the CMake option when the backend was not compiled in — never a "
        "silent fall back to a slower one.");

    deploy.def(
        "read_metadata",
        [](const std::string& path) {
            const Metadata md = Package::read_metadata_file(path);
            py::dict out;
            out["dataset"] = md.provenance.dataset;
            out["software"] = md.provenance.software;
            out["physics"] = md.provenance.physics;
            out["created"] = md.provenance.created;
            out["field_dimensions_m"] = md.geometry.field_dimensions_m;
            if (const auto composition = md.composition) {
                py::list stages, buffers;
                for (const auto& b : composition->buffers) {
                    py::dict entry;
                    entry["name"] = b.name;
                    entry["elements"] = b.elements;
                    entry["dtype"] = std::string(to_string(b.dtype));
                    buffers.append(entry);
                }
                for (const auto& stage : composition->stages) {
                    py::dict entry, reads, writes, launches;
                    entry["name"] = stage.name;
                    entry["invocation"] = std::string(to_string(stage.invocation));
                    for (const auto& p : stage.reads) reads[p.tensor.c_str()] = p.buffer;
                    for (const auto& p : stage.writes) writes[p.tensor.c_str()] = p.buffer;
                    for (const auto& [ref, launch] : stage.launches) {
                        py::dict l;
                        l["entry_point"] = launch.entry_point;
                        l["target_arch"] = ref.target_arch;
                        l["block_size"] = launch.block_size;
                        l["grid_size"] = launch.grid_size;
                        l["shared_memory_bytes"] = launch.shared_memory_bytes;
                        launches[ref.kind.c_str()] = l;
                    }
                    py::dict weights;
                    weights["elements"] = stage.weights.elements;
                    weights["dtype"] = std::string(to_string(stage.weights.dtype));
                    weights["onnx_external_file"] = stage.weights.onnx_external_file;
                    entry["weights"] = weights;
                    entry["reads"] = reads;
                    entry["writes"] = writes;
                    entry["launches"] = launches;
                    stages.append(entry);
                }
                py::dict wiring;
                wiring["buffers"] = buffers;
                wiring["stages"] = stages;
                out["composition"] = wiring;
            } else {
                out["composition"] = py::none();
            }
            if (md.geometry.voxelization) {
                // Named, like every other entry here: a bare pair leaves the reader to remember
                // which half is counts and which is metres.
                py::dict voxelization;
                voxelization["voxel_counts"] = md.geometry.voxelization->voxel_counts;
                voxelization["voxel_dimensions_m"] = md.geometry.voxelization->voxel_dimensions_m;
                out["voxelization"] = voxelization;
            }
            else
                out["voxelization"] = py::none();
            py::list blocks;
            for (const auto& b : md.blocks) {
                py::dict entry;
                entry["kind"] = std::string(b.kind.get_name());
                entry["known_kind"] = b.kind.is_known();
                entry["name"] = b.name;
                entry["target_arch"] = b.target_arch;
                entry["payload_bytes"] = b.payload_bytes;
                blocks.append(entry);
            }
            out["blocks"] = blocks;
            out["metrics"] = md.metrics;
            py::list tensors;
            for (const auto& d : md.io) tensors.append(tensor_dict(d));
            out["tensors"] = tensors;
            return out;
        },
        py::arg("path"),
        "Read a package's metadata header as a dict. Decodes no graph and creates no inference session, so "
        "listing a directory of models costs a few kilobytes per file whatever the payloads weigh.");
}
