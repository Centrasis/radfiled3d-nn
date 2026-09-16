# radfiled3d-nn

**RF3M** — a deployment container and GPU inference runtime for *neural radiation fields*. C++20.

[RadFiled3D](https://github.com/Centrasis/RadFiled3D) stores a simulated radiation field as a static
binary file (`.rf3`). This project extends that idea to a **dynamic** binary field: a neural network
trained on `.rf3` data, shipped as one self-contained `.rf3m` file, that generates a field on demand
for any beam configuration inside the range it was trained on.

```text
.rf3m ──load──▶ GPU-resident model ──infer──▶ engine-owned Vulkan / D3D12 buffer ──▶ shader
                                         └──▶ GPUCartesianRadiationField ──▶ .rf3
```

The demanding consumer comes first: a 3D engine (Unreal Engine 5) wants the predicted field in its
own graphics memory, per frame, as the user moves the beam. A C ABI for prebuilt binaries and
inference straight into externally allocated graphics memory follow from that, and neither is an
add-on.

This repository is the RF3M format **extracted from the training framework
[`radfield3d-nn`](https://github.com/Centrasis/RadField3DNN) and redesigned**. The format is a
contract between a producer and a consumer; reading a file should not require PyTorch, CUDA and a
trainer to be installed.

## Status

Specification plus a working skeleton. The container codec, the descriptor type system, the
RadFiled3D field subclass, the metadata conversion and the C ABI are implemented and tested (28
tests); the execution backends and the graphics interop are declared, option-gated, and not yet
implemented — a disabled backend throws an error naming the option that would enable it, never a
silent fallback. See **[`HANDOFF.md`](HANDOFF.md)** for the work list.

See **[`requirements.md`](requirements.md)** for the full definition and
**[`CLAUDE.md`](CLAUDE.md)** for the working rules.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # fetches and builds RadFiled3D, glm, BLAKE3, GoogleTest
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/write_sample && ./build/rf3m sample.rf3m
```

A local RadFiled3D checkout short-circuits the fetch:

```sh
cmake -S . -B build -DRADFILED3D_SOURCE_DIR=/path/to/RadFiled3D
```

RadFiled3D is a **mandatory** dependency — the library exists to produce and consume its fields, so
it is never an option. The build needs CMake ≥ 3.24 and a C++20 compiler, and nothing else.

With `-DRFNN_WITH_ONNX=ON`, CMake also downloads Microsoft's prebuilt ONNX Runtime
(`-DORT_VERSION=`, default 1.29.0) and links it; the CUDA package is chosen automatically when a
CUDA execution provider was requested. `-DORT_ROOT=<path>` uses an existing installation instead,
which is required for the ROCm and DirectML providers because neither ships as a release archive.

### Options

The optional dependencies are exactly the accelerator backends, and all of them are off by default:

| Option | Brings in |
| --- | --- |
| `RFNN_WITH_ONNX` | ONNX Runtime core — **fetched on demand** by CMake, no SDK install needed |
| `RFNN_WITH_CUDA` / `RFNN_WITH_TENSORRT` | NVIDIA execution providers |
| `RFNN_WITH_ROCM` | AMD (ROCm / MIGraphX) |
| `RFNN_WITH_DIRECTML` | Windows, vendor-neutral |
| `RFNN_WITH_VULKAN` / `RFNN_WITH_DX11` / `RFNN_WITH_DX12` | graphics memory interop — one option each, independent of the compute backend and of each other; none needs a graphics SDK |
| `RFNN_WITH_CPU` | the portable execution provider (ON by default) |
| `RFNN_BUILD_PYTHON` | the pybind11 extension module — **on by default** for a top-level build when Python headers are found |

### Targets

| Target | What it is |
| --- | --- |
| `rfnn::deploy` | the `.rf3m` container (`RadFiled3D::nn::deploy`) — links BLAKE3 and nothing else |
| `rfnn::rfnn` | fields, sessions, memory interop, backends (`RadFiled3D::nn`) — links RadFiled3D |
| `rfnn::c` | the C ABI as a self-contained shared library; the only installed target |
| `rf3m` | inspect a package from the command line |

## C++

From source, add the repository and link `rfnn::rfnn`:

```cpp
#include <RadFiled3D/nn.hpp>
using namespace RadFiled3D::nn;

deploy::Package package = deploy::Package::read_file("model.rf3m");
for (const auto* out : package.get_outputs())
    std::cout << out->name << " " << out->semantic.get_name() << "\n";

// A real RadFiled3D field: RadFiled3D::GPUCartesianRadiationField, declared beside
// CartesianRadiationField because FieldStore::store accepts nothing less.
auto gpu = allocate_gpu_field(FieldGeometry::cubic(64, 1.0f));
store_host_field(gpu->to_host_field(), "prediction.rf3", {});
```

A buffer never crosses as a native handle. Everything in `RadFiled3D::nn` takes and returns a
`memory::MemoryRef`, so the same call binds a `std::vector`, an engine-owned D3D12 resource or a
CUDA allocation, and no public header names a GPU type:

```cpp
session->bind_output("flux", memory::host::MemoryRef::of(std::span(voxels)));
session->bind_output("flux", std::make_shared<memory::dx12::MemoryRef>(resource, bytes));
```

Authoring a package — every declared tensor is a name, a semantic, a shape and how to normalise it;
a quantity this library has never heard of is an ordinary `output` call:

```cpp
deploy::PackageBuilder b;
b.provenance("xray-scatter-v3", "radfield3d-nn 2.1", "G4EmStandardPhysics_option4")
 .field_dimensions_m({1.f, 1.f, 1.f})
 .input("position", deploy::Semantic::Position, {3}).unit("m").done()
 .input("beam_direction", deploy::Semantic::BeamDirection, {2}).unit("rad")
     .range(deploy::MinMax{-3.14159, 3.14159}).normalizer(deploy::LinearSym{-3.14159, 3.14159}).done()
 .output("flux", deploy::Semantic::Flux, {1}).normalizer(deploy::LogScale{1e-12, 30.0}).done()
 .output("direction_distribution", deploy::Semantic::from_name("direction_distribution"), {16, 32}).done()
 .graph("trunk", onnx_bytes)
 .metric("air_kerma_smape", 0.043);
b.build().write_file("model.rf3m");
```

## Unreal Engine / prebuilt binaries

A plugin built with the engine's toolchain links the C ABI: `include/RadFiled3D/nn/c_api.h`, with
`include/RadFiled3D/nn/c_api.hpp` as a header-only C++20 RAII wrapper over it. `cmake --install` places the
shared library, the headers and a CMake package (`find_package(rfnn)` → `rfnn::c`).

```cpp
#include <RadFiled3D/nn/c_api.hpp>

RadFiled3D::nn::c::Metadata model("model.rf3m");
for (const auto& name : model.output_names()) { /* ... */ }
```

## Python

pybind11 + scikit-build-core, the same stack RadFiled3D uses:

```sh
pip install .                                                        # CPU-only reader + PackageBuilder
pip install --config-settings=cmake.define.RFNN_WITH_ONNX=ON .       # with the ONNX Runtime vendored
pip install '.[torch]'                                               # + PyTorch export helpers
```

The wheel is self-contained: it carries the ONNX Runtime beside the extension module, which has an
`$ORIGIN` RPATH, so an installed wheel needs no system ORT and no `LD_LIBRARY_PATH`. The `.pyi`
stub is generated from the built module by `pybind11-stubgen`, never hand-written.

Exporting a trained PyTorch module straight into a package:

```python
from RadFiled3D.nn import deploy
from RadFiled3D.nn.torch_export import add_graph, verify_declarations

pkg = deploy.PackageBuilder(dataset="xray-scatter-v3", software="radfield3d-nn 2.1")
pkg.set_field_dimensions([1.0, 1.0, 1.0])
pkg.add_input("position", "position", [3], unit="m")
pkg.add_input("beam_direction", "beam_direction", [2], unit="rad",
              range=(-3.14159, 3.14159),
              normalizer="linear-1_1", normalizer_params=[-3.14159, 3.14159])
pkg.add_output("flux", "flux", [1], normalizer="log_scale", normalizer_params=[1e-12, 30.0])

graph = add_graph(pkg, "trunk", model, (position, beam_direction),
                  input_names=["position", "beam_direction"], output_names=["flux"])
verify_declarations(pkg, graph)   # every declared tensor is really in a graph
pkg.add_metric("air_kerma_smape", 0.043)
pkg.write("model.rf3m")
```

A model that factors into a beam encoder and a trunk is two `add_graph` calls. `torch` is optional
and imported lazily — reading a package never pulls it in.

## License

MIT — see [LICENSE](LICENSE).
