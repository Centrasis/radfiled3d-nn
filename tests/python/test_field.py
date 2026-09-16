"""The Python `GPUCartesianRadiationField`.

The field's reason to exist is that it IS a RadFiled3D field, so what these check is the crossing:
a layer view aliases the voxels rather than copying them, `to_host_field()` hands back RadFiled3D's
OWN `CartesianRadiationField` through pybind11's shared type registry, and `store_rf3()` writes a
real `.rf3` without the RadFiled3D Python module being installed at all.

Everything that needs RadFiled3D's module skips when it is absent; everything else runs anywhere.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from RadFiled3D import nn
from RadFiled3D.nn import memory

rf3 = pytest.importorskip(
    "RadFiled3D.RadFiled3D", reason="the RadFiled3D Python module is not installed"
)


def field(counts=(4, 5, 6), box=(1.0, 2.0, 3.0), *, filled=True):
    f = nn.GPUCartesianRadiationField(list(counts), list(box))
    f.add_channel("prediction")
    f.add_layer("prediction", "flux", "eV")
    if filled:
        view = f.layer("prediction", "flux")
        view[:] = np.arange(view.size, dtype=np.float32)
    return f


# ── geometry ─────────────────────────────────────────────────────────────────────────────────────


def test_the_voxel_size_follows_from_the_box_and_the_resolution():
    f = field(counts=(4, 5, 6), box=(1.0, 2.0, 3.0))
    assert f.layer("prediction", "flux").size == 4 * 5 * 6
    assert tuple(f.voxel_counts) == (4, 5, 6)
    assert tuple(f.field_dimensions_m) == (1.0, 2.0, 3.0)
    assert tuple(f.voxel_dimensions_m) == pytest.approx((0.25, 0.4, 0.5))


@pytest.mark.parametrize("counts", [(0, 4, 4), (4, -1, 4)])
def test_a_degenerate_resolution_is_refused(counts):
    with pytest.raises((ValueError, TypeError)):
        nn.GPUCartesianRadiationField(list(counts), [1.0, 1.0, 1.0])


# ── the layer view aliases the field ─────────────────────────────────────────────────────────────


def test_a_layer_view_writes_through_to_the_voxels():
    f = field(filled=False)
    f.layer("prediction", "flux")[7] = 42.0
    # A second, independently obtained view must see it: both must be the same memory.
    assert f.layer("prediction", "flux")[7] == 42.0


def test_a_layer_view_keeps_the_field_alive():
    view = field().layer("prediction", "flux")  # the field itself is now unreferenced
    assert view.base is not None
    assert view[3] == 3.0


def test_an_unknown_channel_or_layer_is_a_key_error():
    f = field()
    with pytest.raises(KeyError):
        f.layer("nope", "flux")
    with pytest.raises(KeyError):
        f.layer("prediction", "nope")


# ── the crossing into RadFiled3D ─────────────────────────────────────────────────────────────────


def test_to_host_field_returns_radfiled3ds_own_type():
    host = field().to_host_field()
    # Not a wrapper this module defines: RadFiled3D registered the class, and pybind11's shared
    # registry is what lets the cast find it. A mismatched pybind11 fails exactly here.
    assert isinstance(host, rf3.CartesianRadiationField)
    assert type(host).__module__ == "RadFiled3D.RadFiled3D"


def test_to_host_field_carries_the_values():
    f = field()
    voxels = np.asarray(f.layer("prediction", "flux"))
    host = f.to_host_field().get_channel("prediction").get_layer_as_ndarray("flux")
    # RadFiled3D presents the buffer as (nx, ny, nz, 1) with x fastest; the bytes are the same ones.
    assert np.array_equal(np.asarray(host).ravel(order="F"), voxels)


def test_to_host_field_is_a_copy_not_a_view():
    f = field()
    host = f.to_host_field()
    f.layer("prediction", "flux")[0] = -1.0
    assert np.asarray(host.get_channel("prediction").get_layer_as_ndarray("flux")).ravel()[0] == 0.0


def in_a_clean_process(body):
    """Run `body` where RadFiled3D has NOT been imported.

    `importorskip` at module scope registers the type before any test body runs, so an in-process
    assertion about "without importing RadFiled3D first" cannot fail for the reason it names. A
    subprocess starts with an empty registry, which is the only place the claim is testable.
    """
    source = ("import sys\n"
              "from RadFiled3D import nn\n"
              "from RadFiled3D.nn import deploy, inference, memory\n") + textwrap.dedent(body)
    done = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def test_to_host_field_imports_the_native_submodule_itself():
    # The type is registered by RadFiled3D.RadFiled3D, not by the package `__init__`. The binding
    # imports it so a caller who never mentions RadFiled3D still gets the real class back.
    out = in_a_clean_process(
        """
        assert "RadFiled3D.RadFiled3D" not in sys.modules, "something imported it for us"
        f = nn.GPUCartesianRadiationField([2, 2, 2], [1.0, 1.0, 1.0])
        f.add_channel("prediction")
        f.add_layer("prediction", "flux")
        host = f.to_host_field()
        # Proof that the BINDING did the import, not something else in the process.
        assert "RadFiled3D.RadFiled3D" in sys.modules, "nobody imported it"
        print(type(host).__module__)
        """
    )
    assert out == "RadFiled3D.RadFiled3D"


# ── the self-contained path ──────────────────────────────────────────────────────────────────────


def test_store_rf3_needs_no_radfiled3d_python_module(tmp_path):
    path = tmp_path / "standalone.rf3"
    out = in_a_clean_process(
        f"""
        f = nn.GPUCartesianRadiationField([2, 2, 2], [1.0, 1.0, 1.0])
        f.add_channel("prediction")
        f.add_layer("prediction", "flux")
        f.store_rf3({str(path)!r})
        assert "RadFiled3D.RadFiled3D" not in sys.modules, "store_rf3 pulled the module in"
        print("ok")
        """
    )
    assert out == "ok"
    assert path.stat().st_size > 0


def test_store_rf3_writes_a_file_radfiled3d_reads_back(tmp_path):
    path = tmp_path / "field.rf3"
    f = field()
    f.store_rf3(str(path), software="radfiled3d-nn", version="0.1.0")
    assert path.stat().st_size > 0

    loaded = rf3.FieldStore.load(str(path))
    values = np.asarray(loaded.get_channel("prediction").get_layer_as_ndarray("flux"))
    assert np.array_equal(values.ravel(order="F"), np.asarray(f.layer("prediction", "flux")))


# ── the GPU mirror ───────────────────────────────────────────────────────────────────────────────


def test_a_layer_has_no_device_mirror_until_one_is_attached():
    assert field().get_device_memory("prediction", "flux") is None


def test_a_host_buffer_can_stand_in_as_the_mirror():
    f = field()
    backing = np.zeros(f.layer("prediction", "flux").size, np.float32)
    f.set_device_memory("prediction", "flux", memory.Memory(backing))
    mirror = f.get_device_memory("prediction", "flux")
    assert mirror is not None
    assert mirror.size_bytes == backing.nbytes
