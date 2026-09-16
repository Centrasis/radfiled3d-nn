"""The Python inference API.

Buffers are bound by pointer, so what these check is mostly that nothing gets copied behind the
caller's back and that the ownership rules hold. Everything skips cleanly without a model, without
ONNX Runtime, or without a GPU — a machine that only reads packages still runs the suite.
"""

import os
import pathlib

import numpy as np
import pytest

from RadFiled3D.nn import inference, memory

MODEL = os.environ.get(
    "RFNN_TEST_MODEL",
    "/mnt/data/models_nn/logs/pbrf-ds03-3779-noroi/pbrf-ds03-3779-noroi/models/PBRFNet.rf3m",
)

pytestmark = pytest.mark.skipif(
    not pathlib.Path(MODEL).exists(), reason=f"no model at {MODEL}"
)


def load(backend="cpu", device=-1):
    try:
        return inference.load_rf3m(MODEL, backend=backend, device=device)
    except ValueError as err:
        if "feature" in str(err) or "compiled without" in str(err):
            pytest.skip(f"backend {backend} not compiled in: {err}")
        raise


def inputs_for(queries):
    return {
        "position": np.full(queries * 3, 0.5, np.float32),
        "latent": np.full(queries * 192, 0.01, np.float32),
        "region_state": np.zeros(14, np.float32),
    }


def test_a_numpy_array_is_written_in_place():
    session = load()
    assert session.backend == "cpu"
    queries = 4 * 4 * 4
    session.set_voxel_grid([4, 4, 4])

    flux = np.zeros(queries, np.float32)
    spectrum = np.zeros(queries * 32, np.float32)
    for name, array in inputs_for(queries).items():
        session.bind_input(name, array)
    session.bind_output("flux", flux)
    session.bind_output("spectrum", spectrum)
    session.infer()

    # The array the caller still holds is the one that was written — no copy came back.
    assert np.isfinite(flux).all()
    assert np.count_nonzero(flux) > 0


def test_the_same_buffers_are_reused_across_runs():
    """Bindings are registered once and re-read, so editing an input in place is picked up."""
    session = load()
    queries = 2 * 2 * 2
    session.set_voxel_grid([2, 2, 2])
    arrays = inputs_for(queries)
    flux = np.zeros(queries, np.float32)
    for name, array in arrays.items():
        session.bind_input(name, array)
    session.bind_output("flux", flux)
    session.bind_output("spectrum", np.zeros(queries * 32, np.float32))

    session.infer()
    first = flux.copy()
    # Change an input THROUGH THE SAME BUFFER and run again; no rebinding.
    arrays["latent"][:] = 0.05
    session.infer()
    assert not np.array_equal(first, flux), "editing a bound input in place had no effect"


def test_a_non_contiguous_array_is_refused():
    """A copy would be a different buffer, so it raises instead of silently making one."""
    session = load()
    session.set_voxel_grid([2, 2, 2])
    strided = np.zeros((8, 2), np.float32)[:, 0]
    assert not strided.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="contiguous"):
        session.bind_output("flux", strided)


def test_a_wrong_dtype_is_refused():
    session = load()
    session.set_voxel_grid([2, 2, 2])
    with pytest.raises(ValueError, match="float32"):
        session.bind_output("flux", np.zeros(8, np.float64))


def test_a_missing_binding_names_every_one():
    session = load()
    session.set_voxel_grid([2, 2, 2])
    with pytest.raises(ValueError) as caught:
        session.infer()
    message = str(caught.value)
    for name in ("position", "latent", "flux", "spectrum"):
        assert name in message, f"{name} missing from: {message}"


def test_changing_the_grid_clears_the_bindings():
    session = load()
    session.set_voxel_grid([2, 2, 2])
    session.bind_output("flux", np.zeros(8, np.float32))
    # The shapes depended on the old grid, so the binding cannot survive it.
    session.set_voxel_grid([4, 4, 4])
    with pytest.raises(ValueError, match="flux"):
        session.infer()


def test_memory_reports_where_a_buffer_lives():
    host = memory.Memory(np.zeros(16, np.float32))
    assert host.domain == "host"
    assert host.size_bytes == 64
    assert "host" in repr(host)


def test_an_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="cpu, cuda, tensorrt"):
        inference.load_rf3m(MODEL, backend="gpu")


def test_a_torch_cuda_tensor_binds_as_device_memory():
    """The point of the whole design: a GPU tensor stays on the GPU."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    session = load(backend="cuda", device=0)

    queries = 4 * 4 * 4
    session.set_voxel_grid([4, 4, 4])
    tensors = {
        "position": torch.full((queries * 3,), 0.5, dtype=torch.float32, device="cuda"),
        "latent": torch.full((queries * 192,), 0.01, dtype=torch.float32, device="cuda"),
        "region_state": torch.zeros(14, dtype=torch.float32, device="cuda"),
    }
    flux = torch.zeros(queries, dtype=torch.float32, device="cuda")
    spectrum = torch.zeros(queries * 32, dtype=torch.float32, device="cuda")

    assert memory.Memory(flux).domain == "cuda", "a CUDA tensor must bind as device memory"
    for name, tensor in tensors.items():
        session.bind_input(name, tensor)
    session.bind_output("flux", flux)
    session.bind_output("spectrum", spectrum)
    session.infer()

    torch.cuda.synchronize()
    assert torch.isfinite(flux).all()
    assert int(torch.count_nonzero(flux)) > 0


def test_a_cpu_torch_tensor_binds_as_host_memory():
    """Recognised by duck-typing, so the extension never imports torch."""
    torch = pytest.importorskip("torch")
    tensor = torch.zeros(16, dtype=torch.float32)
    buffer = memory.Memory(tensor)
    assert buffer.domain == "host"
    assert buffer.size_bytes == 64
