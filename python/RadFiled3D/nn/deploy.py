"""``RadFiled3D.nn.deploy`` — the .rf3m container, mirroring C++ ``RadFiled3D::nn::deploy``.

Describe a model, write it, read it back. Nothing here needs a GPU or a runtime: reading a package
is parsing bytes, which is the separation the C++ target ``rfnn::deploy`` enforces by linking BLAKE3
and nothing else.

A real module rather than an attribute of the extension, so ``import RadFiled3D.nn.deploy`` works —
pybind11's ``def_submodule`` creates an attribute and no ``sys.modules`` entry.
"""

from ._rfnn.deploy import PackageBuilder, read_metadata

__all__ = ["PackageBuilder", "read_metadata"]
