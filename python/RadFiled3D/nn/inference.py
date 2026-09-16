"""``RadFiled3D.nn.inference`` — run a package on a device, mirroring C++ ``core/``.

The binding protocol is the one in ``requirements.md`` R-I1::

    set_voxel_grid -> bind_input / bind_output -> infer

The caller owns every buffer and the session allocates no result. Bindings are registered once and
re-read on every run, so editing an input in place is picked up without rebinding.
"""

from ._rfnn.inference import Session, load_rf3m

__all__ = ["Session", "load_rf3m"]
