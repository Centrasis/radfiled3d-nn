# radfiled3d-nn

[![build, test, publish](https://github.com/Centrasis/radfiled3d-nn/actions/workflows/build-test-publish.yml/badge.svg)](https://github.com/Centrasis/radfiled3d-nn/actions/workflows/build-test-publish.yml)
[![PyPI](https://img.shields.io/pypi/v/radfiled3d-nn.svg)](https://pypi.org/project/radfiled3d-nn/)
[![Python versions](https://img.shields.io/pypi/pyversions/radfiled3d-nn.svg)](https://pypi.org/project/radfiled3d-nn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**RF3M** — a deployment container and GPU inference runtime for *neural radiation fields*. C++20.

[RadFiled3D](https://github.com/Centrasis/RadFiled3D) stores a simulated radiation field as a static
binary file (`.rf3`). This project extends that idea to a **dynamic** binary field: a neural network
trained on `.rf3` data, shipped as one self-contained `.rf3m` file, that generates a field on demand
for any beam configuration inside the range it was trained on.

```text
.rf3m ──load──▶ GPU-resident model ──infer──▶ engine-owned Vulkan / D3D12 / D3D11 buffer ──▶ shader
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

The container, the runtime and both foreign surfaces are implemented and tested. CI builds the
default option set, the ONNX Runtime configuration and an ASan/UBSan run on every push; the GPU
configurations are built and tested on developer hardware.

| Area | State |
| --- | --- |
| `.rf3m` container, descriptors, `PackageBuilder`, `rf3m` tool | implemented, tested |
| Multi-stage composition: named buffers, stage ordering, per-stage weights | implemented, tested |
| ONNX Runtime sessions — CPU, CUDA, TensorRT | implemented, tested on hardware |
| CUDA kernel stages (PTX / cubin, driver API) | implemented, tested on hardware |
| Vulkan → CUDA memory import | implemented, verified end to end on an RTX 5090 |
| D3D12 → CUDA / TensorRT, D3D11 → CUDA / TensorRT | implemented; pairing and refusals tested on Linux |
| DirectML (D3D12 as native memory, no import) | written, **never compiled** — Windows-only, needs a Windows build |
| ROCm / MIGraphX (AMD) | declared and option-gated; every operation throws `FeatureDisabled` |
| C ABI + Python module | implemented, tested |

A disabled backend throws an error naming the CMake option that would enable it — never a silent
fallback to a slower path.

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
| `RFNN_WITH_CUDA` | the CUDA execution provider, and CUDA kernel stages through the driver API |
| `RFNN_WITH_TENSORRT` | the TensorRT execution provider (same hardware family as CUDA) |
| `RFNN_WITH_ROCM` | the ROCm / MIGraphX execution provider (AMD) |
| `RFNN_WITH_DIRECTML` | the DirectML execution provider (Windows, vendor-neutral) |
| `RFNN_WITH_VULKAN` | Vulkan external-memory interop |
| `RFNN_WITH_DX11` | Direct3D 11 external-memory interop |
| `RFNN_WITH_DX12` | Direct3D 12 external-memory interop |
| `RFNN_WITH_CPU` | the portable execution provider (**ON** by default) |
| `RFNN_BUILD_PYTHON` | the pybind11 extension module — **on** by default for a top-level build when Python headers are found |

`RFNN_BUILD_TESTS`, `RFNN_BUILD_TOOLS`, `RFNN_BUILD_EXAMPLES` and `RFNN_INSTALL` default to on for a
top-level build and off when the project is added as a subdirectory.

### Graphics interop

**Interop is a pair, and the two halves are chosen independently.** Sharing memory takes a graphics
API on one side and a compute backend on the other, so no graphics option implies a compute one:

| graphics ↓ / compute → | CUDA (NVIDIA) | ROCm / HIP (AMD) | DirectML | CPU |
| --- | --- | --- | --- | --- |
| **Vulkan** | external memory import | external memory import | ✗ | ✗ |
| **D3D12** | external memory import | external memory import | native — no import | ✗ |
| **D3D11** | external memory import | external memory import | ✗ | ✗ |

Two things follow. Vulkan and D3D12 memory is shareable from an AMD card through HIP exactly as
from an NVIDIA card through CUDA. And under the DirectML provider a D3D12 resource already *is* the
provider's own memory, so the import is a no-op — the fast path on Windows needs no interop layer.

**Graphics interop needs no graphics SDK.** Importing a Vulkan allocation or a D3D12 resource is an
import call on a handle the renderer already exported, so none of `RFNN_WITH_VULKAN`,
`RFNN_WITH_DX11` or `RFNN_WITH_DX12` links a loader or an SDK. Which pairing is in play is decided
at **runtime**; an impossible pairing is `UnsupportedInterop`, which is a different error from a
backend that merely was not compiled in.

**D3D11 and D3D12 stay apart** rather than sharing a "DirectX" option: D3D12 imports once through
external memory, D3D11 registers and maps per use.

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

### Choosing the device

A session runs on **one** device, and the caller chooses it. The graphics flow is import first, then
take the device from the imported buffer — the renderer already decided which GPU its memory is on:

```cpp
memory::vk::ExternalMemory interop;                       // or memory::dx12:: / memory::dx11::
auto imported = interop.import_buffer(exported, Backend::Cuda);
auto session = load(package, Backend::Cuda, Device::of(*imported));   // or Device::ordinal(1)
```

The resolved device is used for everything: the execution provider, the buffers between stages, a
kernel stage's module, and its weights. Memory from another device is refused at bind, naming both
indices.

### Several stages, one package

A model may be more than one graph. The package records the **named buffers** between stages, the
stages in execution order, and whether each runs `Once` per field or `PerQuery` — so a beam encoder
whose result every voxel reuses runs once. A stage may read a buffer written by any strictly
earlier stage; buffers are allocated when the grid is chosen and reused by every inference.

A stage is an ONNX graph *or* a compiled kernel, and both are ordinary blocks in the container. One
implementation is enough, and it need not be the portable one: a package carrying only
`cuda_ptx:hashgrid` refuses unsupported hardware up front, naming every stage responsible, rather
than failing somewhere in the middle of a frame.

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

The ABI carries the whole inference protocol — importing Vulkan, D3D12 and D3D11 memory included —
and nothing throws across it; every entry point returns a status code.

## Python

pybind11 + scikit-build-core, the same stack RadFiled3D uses. **The wheel extends RadFiled3D**: the
distribution is `radfiled3d-nn` and it installs into RadFiled3D's own package directory, so
`from RadFiled3D import nn` sits beside `from RadFiled3D import RadFiled3D` — the mirror of
`<RadFiled3D/nn/deploy.hpp>` beside `<RadFiled3D/RadiationField.hpp>`.

```sh
pip install radfiled3d-nn          # the released wheel: ONNX Runtime vendored, no system ORT needed
pip install '.[torch]'             # from source, + the PyTorch export helpers
```

The submodules mirror the C++ namespaces:

| Python | C++ | holds |
| --- | --- | --- |
| `RadFiled3D.nn.deploy` | `nn::deploy` | `PackageBuilder`, `read_metadata` |
| `RadFiled3D.nn.inference` | `nn` (`core/`) | `Session`, `load_rf3m` |
| `RadFiled3D.nn.memory` | `nn::memory` | `Memory` |
| `RadFiled3D.nn` | `nn` | `version()`, `GPUCartesianRadiationField` |

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

Running one — a buffer is bound by pointer, and a torch CUDA tensor never leaves the GPU:

```python
from RadFiled3D.nn.inference import load_rf3m

session = load_rf3m("model.rf3m", backend="cuda")
session.bind_input("beam_direction", beam)      # numpy array or torch tensor
session.bind_output("flux", out)                # a torch CUDA tensor binds as device memory
session.infer()
```

`torch` is optional and imported lazily — reading a package never pulls it in, and the extension
recognises a tensor by duck-typing rather than by importing torch.

## License

MIT — see [LICENSE](LICENSE).
