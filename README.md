# radfiled3d-nn

[![build](https://github.com/Centrasis/radfiled3d-nn/actions/workflows/build.yml/badge.svg)](https://github.com/Centrasis/radfiled3d-nn/actions/workflows/build.yml)
[![release](https://github.com/Centrasis/radfiled3d-nn/actions/workflows/release.yml/badge.svg)](https://github.com/Centrasis/radfiled3d-nn/actions/workflows/release.yml)
[![PyPI](https://img.shields.io/pypi/v/radfiled3d-nn.svg)](https://pypi.org/project/radfiled3d-nn/)
[![Python versions](https://img.shields.io/pypi/pyversions/radfiled3d-nn.svg)](https://pypi.org/project/radfiled3d-nn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**RF3M** — a deployment container and GPU inference runtime for *neural radiation fields*. C++20,
with a Python module and a C ABI.

[RadFiled3D](https://github.com/Centrasis/RadFiled3D) stores a simulated radiation field as a static
file (`.rf3`). This project ships the **dynamic** version: a network trained on that data, packaged
as one self-contained `.rf3m` file, that generates a field on demand for any beam configuration it
was trained on.

```text
.rf3m ──load──▶ GPU-resident model ──infer──▶ engine-owned Vulkan / D3D12 / D3D11 buffer ──▶ shader
                                         └──▶ GPUCartesianRadiationField ──▶ .rf3
```

## Scope

**In scope:** the `.rf3m` format (read, write, inspect), running a packaged model on CPU, CUDA or
TensorRT, and writing the result into memory a 3D engine already owns.

**Out of scope:** training. Models are trained in
[`radfield3d-nn`](https://github.com/Centrasis/RadField3DNN), which exports through this library's
`PackageBuilder`. Nothing here depends on PyTorch.

## What to expect

- **One file carries everything** — graphs or kernels, weights, the tensor contract, normalisation,
  provenance and the wiring between stages.
- **Reading needs no GPU.** Parsing is separate from execution; inspecting a package pulls in no
  ONNX Runtime and no CUDA.
- **A disabled backend throws**, naming the CMake option that would enable it. There is never a
  silent fallback to a slower path.
- **Buffers bind by pointer**, including engine memory — a NumPy array, a torch CUDA tensor or an
  imported Vulkan/D3D12 allocation all bind the same way, with no host round-trip.
- **A session runs on one device**, chosen by the caller. Memory from another device is refused at
  bind rather than faulting.
- **The default build is CPU-only.** Every accelerator is an opt-in CMake option.

## Install

```sh
pip install radfiled3d-nn          # extends the RadFiled3D package: `from RadFiled3D import nn`
```

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # fetches RadFiled3D, glm, BLAKE3, GoogleTest
cmake --build build -j
```

Accelerators are off by default: `-DRFNN_WITH_ONNX=ON` (fetches the ONNX Runtime), then any of
`RFNN_WITH_CUDA`, `_TENSORRT`, `_ROCM`, `_DIRECTML` for compute and `RFNN_WITH_VULKAN`, `_DX11`,
`_DX12` for graphics-memory interop. The two halves are independent — no graphics option implies a
compute one, and neither needs a graphics SDK installed.

## Python

Inspect a package:

```python
from RadFiled3D.nn import deploy

md = deploy.read_metadata("model.rf3m")          # reads a few hundred bytes, not the payloads
print(md["dataset"], md["field_dimensions_m"])
for b in md["blocks"]:
    print(b["kind"], b["name"], b["payload_bytes"])
```

Run it:

```python
from RadFiled3D.nn.inference import load_rf3m

session = load_rf3m("model.rf3m", backend="cuda")
session.bind_input("beam_direction", beam)   # numpy array or torch tensor
session.bind_output("flux", out)             # a torch CUDA tensor stays on the GPU
session.infer()
```

Author one:

```python
pkg = deploy.PackageBuilder(dataset="xray-scatter-v3", software="radfield3d-nn 2.1")
pkg.set_field_dimensions([1.0, 1.0, 1.0])
pkg.add_input("position", "position", [3], unit="m")
pkg.add_output("flux", "flux", [1], normalizer="log_scale", normalizer_params=[1e-12, 30.0])
pkg.add_graph("trunk", onnx_bytes)
pkg.write("model.rf3m")
```

`RadFiled3D.nn.torch_export` turns a `torch.nn.Module` into those bytes; `torch` is optional and
imported lazily.

## C++

```cpp
#include <RadFiled3D/nn.hpp>
using namespace RadFiled3D::nn;

auto package = deploy::Package::read_file("model.rf3m");
for (const auto* out : package.get_outputs())
    std::cout << out->name << " " << out->semantic.get_name() << "\n";
```

```cpp
auto session = load(package, Backend::Cuda, Device::ordinal(0));
session->bind_output("flux", memory::host::MemoryRef::of(std::span(voxels)));
session->infer();
```

Into a renderer's own memory — import first, then run on the device that buffer is already on:

```cpp
memory::vk::ExternalMemory interop;                   // or memory::dx12:: / memory::dx11::
auto imported = interop.import_buffer(exported, Backend::Cuda);
auto session  = load(package, Backend::Cuda, Device::of(*imported));
```

Link `rfnn::rfnn` from source, or `rfnn::c` for the C ABI (`<RadFiled3D/nn/c_api.h>`, plus a
header-only RAII wrapper) when the consumer is a prebuilt plugin such as UE5.

## The `.rf3m` format, version 1

Little-endian throughout. `[str]` is `[u32 len][utf-8 bytes]`, not NUL-terminated.

```text
[4]    magic             "RF3M"
[u32]  version           1
[32]   digest            BLAKE3 of every byte after this field
[u64]  metadata_bytes    skip this many to reach the first block
[...]  metadata          provenance, I/O descriptors, geometry, metrics, RadFiled3D provenance
[u32]  block_count
  per block:
    [u64] block_bytes    everything after this field, so an unwanted block is one seek
    [str] kind           "onnx", "cuda_ptx", "cuda_cubin", "hsaco", "spirv", "opencl",
                         "weights", "composition"
    [str] name           "trunk", "beam_encoder", ...
    [str] target_arch    "sm_90"; empty = runs anywhere its kind runs
    [...] payload
```

Four properties matter to anyone building on it:

- **Skip what you did not ask for.** Listing a 400 MB package reads a few hundred bytes — the
  header, one seek, then `[len][kind][name]` per block.
- **Graphs and kernels are one concept.** An ONNX graph and a CUDA kernel differ only by `kind`, so
  a package may carry both and the loader picks what the hardware can run.
- **Unknown blocks survive a rewrite.** An older tool re-saving a package never strips the code for
  an accelerator it has not heard of.
- **Integrity is checked before parsing.** The digest catches a truncated or edited file up front.

A model may be several stages. A `composition` block names the buffers between them, their order,
and whether each runs once per field or once per query — so a beam encoder whose result every voxel
reuses runs once. Stage weights travel in a `weights` block sharing the stage's name.

## License

MIT — see [LICENSE](LICENSE).
