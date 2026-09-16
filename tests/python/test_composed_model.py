"""A model shaped like a real one, written and read from Python, then run.

This is the end-to-end case the flat round-trip tests do not reach: several stages, an encoder whose
result is computed ONCE per field and reused at every voxel — PBRFNet's global beam parameters — and
a trunk with THREE outputs. The graphs are built here with ``onnx`` rather than loaded from a fixture
so the test runs anywhere the exporter does, CI included.

The arithmetic is deliberately trivial and every expected value is derivable by hand, which is what
makes a wrong voxel order or an un-repeated latent fail on the numbers instead of passing a
"something was written" check:

    beam_encoder(direction[3], distance[1])   -> latent[1, 2] = [s, s * d],  s = sum(direction)
    trunk(position[q, 3], latent[q, 2])       -> flux[q, 1]     = sum(position) + latent0 + latent1
                                                 dose[q, 1]     = flux * 2
                                                 spectrum[q, 4] = concat(position, flux)
"""

import numpy as np
import pytest

from RadFiled3D.nn import deploy, inference

onnx = pytest.importorskip("onnx", reason="building the graphs needs onnx")
from onnx import TensorProto, helper  # noqa: E402

QUERIES = "queries"
LATENT = 2
SPECTRUM = 4


def _model(nodes, inputs, outputs, initializers, name):
    graph = helper.make_graph(nodes, name, inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)],
                              producer_name="radfiled3d-nn tests")
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model.SerializeToString()


def beam_encoder() -> bytes:
    """Runs ONCE per field: a beam is constant over the volume, so its latent is too."""
    direction = helper.make_tensor_value_info("beam_direction", TensorProto.FLOAT, [3])
    distance = helper.make_tensor_value_info("source_distance", TensorProto.FLOAT, [1])
    latent = helper.make_tensor_value_info("latent", TensorProto.FLOAT, [1, LATENT])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
    nodes = [
        helper.make_node("ReduceSum", ["beam_direction", "axes"], ["s"], keepdims=1),
        helper.make_node("Mul", ["s", "source_distance"], ["sd"]),
        helper.make_node("Concat", ["s", "sd"], ["row"], axis=0),
        helper.make_node("Unsqueeze", ["row", "axes"], ["latent"]),
    ]
    return _model(nodes, [direction, distance], [latent], [axes], "beam_encoder")


def trunk() -> bytes:
    """Runs at EVERY voxel, consuming the latent the encoder computed once."""
    position = helper.make_tensor_value_info("position", TensorProto.FLOAT, [QUERIES, 3])
    latent = helper.make_tensor_value_info("latent", TensorProto.FLOAT, [QUERIES, LATENT])
    flux = helper.make_tensor_value_info("flux", TensorProto.FLOAT, [QUERIES, 1])
    dose = helper.make_tensor_value_info("dose", TensorProto.FLOAT, [QUERIES, 1])
    spectrum = helper.make_tensor_value_info("spectrum", TensorProto.FLOAT, [QUERIES, SPECTRUM])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    nodes = [
        helper.make_node("ReduceSum", ["position", "axes"], ["p"], keepdims=1),
        helper.make_node("ReduceSum", ["latent", "axes"], ["l"], keepdims=1),
        helper.make_node("Add", ["p", "l"], ["flux"]),
        helper.make_node("Mul", ["flux", "two"], ["dose"]),
        helper.make_node("Concat", ["position", "flux"], ["spectrum"], axis=1),
    ]
    return _model(nodes, [position, latent], [flux, dose, spectrum], [axes, two], "trunk")


def build(path) -> str:
    """Describe the whole model and write it, exactly as a trainer's export would."""
    pkg = deploy.PackageBuilder(dataset="DS03-composed", software="radfiled3d-nn tests",
                                physics="none", created="2026-09-16T00:00:00Z")
    pkg.set_field_dimensions([1.0, 1.0, 1.0])

    # What a CALLER supplies. The latent is not here: it is the composition's to compute.
    pkg.add_input("position", "position", [3], unit="m")
    pkg.add_input("beam_direction", "beam_direction", [3], unit="rad")
    pkg.add_input("source_distance", "source_distance", [1], unit="m", range=(0.5, 3.0))
    pkg.add_output("flux", "flux", [1])
    pkg.add_output("dose", "dose", [1], unit="Gy")
    pkg.add_output("spectrum", "spectrum", [SPECTRUM], unit="eV")

    pkg.add_graph("beam_encoder", beam_encoder())
    pkg.add_graph("trunk", trunk())

    # The wiring: one named buffer carrying the encoder's single row to every query.
    pkg.add_buffer("beam_latent", LATENT)
    pkg.add_stage("beam_encoder", "once", writes={"latent": "beam_latent"})
    pkg.add_stage("trunk", "per_query", reads={"latent": "beam_latent"})

    pkg.add_metric("air_kerma_smape", 0.043)
    pkg.write(str(path))
    return str(path)


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    return build(tmp_path_factory.mktemp("composed") / "composed.rf3m")


def test_what_was_written_is_what_is_read_back(package):
    md = deploy.read_metadata(package)
    assert md["dataset"] == "DS03-composed"
    assert md["metrics"]["air_kerma_smape"] == pytest.approx(0.043)

    # Three outputs, and the caller's inputs — the latent appears in neither, because it is internal
    # to the composition and a caller never supplies or sees one.
    by_role = {"input": set(), "output": set()}
    for tensor in md["tensors"]:
        by_role[tensor["role"]].add(tensor["name"])
    assert by_role["output"] == {"flux", "dose", "spectrum"}
    assert by_role["input"] == {"position", "beam_direction", "source_distance"}
    assert "latent" not in by_role["input"] | by_role["output"]

    wiring = md["composition"]
    assert [s["name"] for s in wiring["stages"]] == ["beam_encoder", "trunk"]
    assert [s["invocation"] for s in wiring["stages"]] == ["once", "per_query"]
    assert wiring["buffers"] == [{"name": "beam_latent", "elements": LATENT, "dtype": "f32"}]
    assert wiring["stages"][0]["writes"] == {"latent": "beam_latent"}
    assert wiring["stages"][1]["reads"] == {"latent": "beam_latent"}
    # The blocks a scan sees, without a payload being touched.
    assert {b["name"]: b["kind"] for b in md["blocks"]} == {
        "beam_encoder": "onnx", "trunk": "onnx", "composition": "composition",
    }


def test_the_encoders_result_is_reused_at_every_voxel(package):
    """The whole point of an `once` stage: computed one time, correct at every query."""
    session = inference.load_rf3m(package, backend="cpu")
    side = 2
    queries = side ** 3
    session.set_voxel_grid([side, side, side])

    rng = np.random.default_rng(0)
    position = rng.random(queries * 3, dtype=np.float32)
    direction = np.array([0.25, 0.5, 1.0], np.float32)
    distance = np.array([2.0], np.float32)

    flux = np.zeros(queries, np.float32)
    dose = np.zeros(queries, np.float32)
    spectrum = np.zeros(queries * SPECTRUM, np.float32)

    session.bind_input("position", position)
    session.bind_input("beam_direction", direction)
    session.bind_input("source_distance", distance)
    session.bind_output("flux", flux)
    session.bind_output("dose", dose)
    session.bind_output("spectrum", spectrum)
    session.infer()

    s = float(direction.sum())
    latent_sum = s + s * float(distance[0])
    p = position.reshape(queries, 3).sum(axis=1)

    # Every voxel, not just the first: a latent that was computed but never repeated across the
    # batch gives the right answer for query 0 and zeros after it.
    np.testing.assert_allclose(flux, p + latent_sum, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dose, (p + latent_sum) * 2.0, rtol=1e-5, atol=1e-5)
    written = spectrum.reshape(queries, SPECTRUM)
    np.testing.assert_allclose(written[:, :3], position.reshape(queries, 3), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(written[:, 3], p + latent_sum, rtol=1e-5, atol=1e-5)


def test_the_latent_is_not_the_callers_to_bind(package):
    session = inference.load_rf3m(package, backend="cpu")
    session.set_voxel_grid([1, 1, 1])
    # It is fed from a buffer the encoder writes, and the refusal says so rather than "no such input".
    with pytest.raises(ValueError, match="beam_latent"):
        session.bind_input("latent", np.zeros(LATENT, np.float32))


def test_every_missing_binding_is_named_at_once(package):
    session = inference.load_rf3m(package, backend="cpu")
    session.set_voxel_grid([1, 1, 1])
    session.bind_input("position", np.zeros(3, np.float32))
    with pytest.raises(ValueError) as err:
        session.infer()
    # Finding them one run at a time is what makes driving a model tedious (R-I1).
    message = str(err.value)
    for missing in ("beam_direction", "source_distance", "flux", "dose", "spectrum"):
        assert missing in message, message


def test_a_second_run_reuses_everything(package):
    """Buffers are instantiated with the grid and reused; a repeated inference allocates nothing."""
    session = inference.load_rf3m(package, backend="cpu")
    session.set_voxel_grid([2, 2, 2])
    position = np.full(8 * 3, 0.25, np.float32)
    direction = np.array([1.0, 1.0, 1.0], np.float32)
    distance = np.array([1.0], np.float32)
    first, second = np.zeros(8, np.float32), np.zeros(8, np.float32)
    scratch = np.zeros(8, np.float32)
    spectrum = np.zeros(8 * SPECTRUM, np.float32)

    session.bind_input("position", position)
    session.bind_input("beam_direction", direction)
    session.bind_input("source_distance", distance)
    session.bind_output("dose", scratch)
    session.bind_output("spectrum", spectrum)

    session.bind_output("flux", first)
    session.infer()
    session.bind_output("flux", second)
    session.infer()

    np.testing.assert_array_equal(first, second)
    # 0.25 * 3 + (3 + 3 * 1) = 0.75 + 6
    np.testing.assert_allclose(first, np.full(8, 6.75, np.float32), rtol=1e-5)
