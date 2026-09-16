"""RF3M — the deployment container and GPU inference runtime for neural radiation fields.

This package EXTENDS RadFiled3D. It ships as the distribution ``radfiled3d-nn`` and installs into
``RadFiled3D/nn``, so a consumer writes::

    from RadFiled3D import RadFiled3D as rf3      # the field format
    from RadFiled3D import nn                     # the models trained on it

exactly as C++ puts ``<RadFiled3D/nn/deploy.hpp>`` beside ``<RadFiled3D/RadiationField.hpp>``. The
submodules mirror the C++ namespaces one for one:

    RadFiled3D.nn.deploy      ::nn::deploy  — describe a model, write it, read it back
    RadFiled3D.nn.inference   core/         — run it on a device and bind its tensors
    RadFiled3D.nn.memory      ::nn::memory  — the buffers inference reads and writes
    RadFiled3D.nn             ::nn          — the version, and the fields a prediction fills

The compiled extension ``_rfnn`` is the whole implementation; these modules only re-export it, so
there is exactly one implementation of the byte layout and Python never learns it.

Exporting a trained model::

    from RadFiled3D.nn import deploy

    pkg = deploy.PackageBuilder(dataset="xray-scatter-v3", software="radfield3d-nn 2.1")
    pkg.set_field_dimensions([1.0, 1.0, 1.0])
    pkg.add_input("position", "position", [3], unit="m",
                  normalizer="linear0_1", normalizer_params=[0.0, 1.0])
    pkg.add_output("flux", "flux", [1], normalizer="log_scale", normalizer_params=[1e-12, 30.0])
    pkg.add_graph("trunk", onnx_bytes)
    pkg.write("model.rf3m")

Running one::

    from RadFiled3D.nn import inference

    session = inference.load_rf3m("model.rf3m", backend="cuda")
    session.set_voxel_grid([64, 64, 64])
    session.bind_input("position", positions)      # any contiguous float32 buffer
    session.bind_output("flux", flux)
    session.infer()

Buffers are bound **by pointer and never copied**, so a torch CUDA tensor stays on the GPU for the
whole run and ``flux`` is filled in place. A non-contiguous array raises rather than being quietly
copied — a copy would not be the buffer you asked to have written.
"""

from . import deploy, inference, memory
from ._rfnn import GPUCartesianRadiationField, version

__all__ = [
    "deploy",
    "inference",
    "memory",
    "GPUCartesianRadiationField",
    "version",
    "torch_export",
    "__version__",
]

# ONE version, and the C++ library holds it: `project(radfiled3d_nn VERSION ...)` in CMakeLists.txt
# feeds `version.hpp.in`, which is where `version()` comes from — and the wheel's own version is read
# from that same line at build time (build_tools/radfiled3d_nn_version.py). So this is not a second
# number that could drift; it is the same one, reported through the Python name for it.
__version__ = version()


def __getattr__(name: str):
    """Expose ``RadFiled3D.nn.torch_export`` without importing PyTorch at package import.

    ``torch`` is an optional dependency and an expensive import; a consumer that only reads packages
    must not pay for it. The submodule is loaded on first attribute access instead.
    """
    if name == "torch_export":
        from . import torch_export

        return torch_export
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _cli() -> int:
    """``rf3m <package.rf3m>`` — print what a package declares."""
    import json
    import sys

    if len(sys.argv) != 2:
        print("usage: rf3m <package.rf3m>", file=sys.stderr)
        return 1
    try:
        print(json.dumps(deploy.read_metadata(sys.argv[1]), indent=2, default=str))
    except (OSError, ValueError) as err:
        print(f"rf3m: {sys.argv[1]}: {err}", file=sys.stderr)
        return 1
    return 0
