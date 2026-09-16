"""The PyTorch → ONNX → ``.rf3m`` export path.

Run with the extension installed and the optional torch extra present::

    pip install '.[torch]'          # in the RadSim conda env
    pytest tests/python

Every test skips cleanly without torch, because torch is an optional dependency and a machine that
only reads packages should still be able to run the suite.
"""

import numpy as np
import pytest

from RadFiled3D import nn
from RadFiled3D.nn import deploy

torch = pytest.importorskip("torch")
nn = torch.nn

from RadFiled3D.nn.torch_export import (  # noqa: E402
    add_graph,
    export_to_onnx,
    graph_io_names,
    verify_declarations,
)


class Trunk(nn.Module):
    """A point field: (position, beam_direction) -> flux. The deployable shape — scale-free."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, position, beam_direction):
        return self.net(torch.cat([position, beam_direction], dim=-1))


@pytest.fixture
def model():
    return Trunk().eval()


@pytest.fixture
def example_args():
    return torch.zeros(4, 3), torch.zeros(4, 2)


def declared_builder() -> deploy.PackageBuilder:
    pkg = deploy.PackageBuilder(dataset="xray-scatter-v3", software="radfield3d-nn 2.1")
    pkg.set_field_dimensions([1.0, 1.0, 1.0])
    pkg.add_input("position", "position", [3], unit="m")
    pkg.add_input(
        "beam_direction",
        "beam_direction",
        [2],
        unit="rad",
        range=(-3.14159, 3.14159),
        normalizer="linear-1_1",
        normalizer_params=[-3.14159, 3.14159],
    )
    pkg.add_output("flux", "flux", [1], normalizer="log_scale", normalizer_params=[1e-12, 30.0])
    return pkg


def test_a_module_becomes_a_package(tmp_path, model, example_args):
    pkg = declared_builder()
    onnx_bytes = add_graph(
        pkg,
        "trunk",
        model,
        example_args,
        input_names=["position", "beam_direction"],
        output_names=["flux"],
    )
    pkg.add_metric("air_kerma_smape", 0.043)

    path = tmp_path / "model.rf3m"
    pkg.write(str(path))

    md = deploy.read_metadata(str(path))
    assert [b["name"] for b in md["blocks"]] == ["trunk"]
    assert md["blocks"][0]["kind"] == "onnx"
    assert md["blocks"][0]["payload_bytes"] == len(onnx_bytes)
    assert md["metrics"] == {"air_kerma_smape": 0.043}

    # The names the runtime binds by must be the names the graph actually declares.
    assert graph_io_names(onnx_bytes) == (["position", "beam_direction"], ["flux"])
    declared = {t["name"] for t in md["tensors"]}
    assert declared == {"position", "beam_direction", "flux"}


def test_the_batch_axis_stays_dynamic(model, example_args):
    """Traced at batch 4, the stored graph must still run at any other batch size.

    Without a dynamic axis the exporter freezes the traced batch into the graph and the deployed
    model rejects every other size — which is invisible until deployment.
    """
    ort = pytest.importorskip("onnxruntime")
    onnx_bytes = export_to_onnx(
        model,
        example_args,
        input_names=["position", "beam_direction"],
        output_names=["flux"],
    )
    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    out = session.run(
        None,
        {
            "position": np.zeros((17, 3), np.float32),
            "beam_direction": np.zeros((17, 2), np.float32),
        },
    )
    assert out[0].shape == (17, 1)


def test_a_none_argument_is_refused(model, example_args):
    """`input_names` is positional, and a None argument produces no graph input — so a name listed
    for it lands on the next tensor. Silent, and fatal at deployment."""
    position, _ = example_args
    with pytest.raises(ValueError, match="positionally"):
        export_to_onnx(
            model,
            (position, None),
            input_names=["position", "beam_direction"],
            output_names=["flux"],
        )


def test_mismatched_name_counts_are_refused(model, example_args):
    with pytest.raises(ValueError, match="matched positionally"):
        export_to_onnx(
            model,
            example_args,
            input_names=["position"],
            output_names=["flux"],
        )


def test_a_name_the_graph_does_not_provide_is_caught(model, example_args):
    """The mismatch that actually happens: the package declares `beam_dir`, the graph declares
    `beam_direction`. The runtime binds by name, so this package would load and then fail to bind."""
    pytest.importorskip("onnx")
    pkg = deploy.PackageBuilder()
    pkg.set_field_dimensions([1.0, 1.0, 1.0])
    pkg.add_input("position", "position", [3], unit="m")
    pkg.add_input("beam_dir", "beam_direction", [2], unit="rad")  # the graph calls it beam_direction
    pkg.add_output("flux", "flux", [1])

    onnx_bytes = add_graph(
        pkg,
        "trunk",
        model,
        example_args,
        input_names=["position", "beam_direction"],
        output_names=["flux"],
    )
    with pytest.raises(ValueError, match="beam_dir"):
        verify_declarations(pkg, onnx_bytes)


def test_matching_declarations_pass(model, example_args):
    pytest.importorskip("onnx")
    pkg = declared_builder()
    onnx_bytes = add_graph(
        pkg,
        "trunk",
        model,
        example_args,
        input_names=["position", "beam_direction"],
        output_names=["flux"],
    )
    verify_declarations(pkg, onnx_bytes)  # does not raise
