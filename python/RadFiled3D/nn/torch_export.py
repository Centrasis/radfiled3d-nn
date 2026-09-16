"""Export a PyTorch module to ONNX for storing in an ``.rf3m`` package.

The ONNX export lives here, in Python, and not in the C++ extension: the exporter is
``torch.onnx``, so it needs PyTorch, and PyTorch has no business being a dependency of a module
whose job is to *read* deployment packages on machines that will never train anything. The extension
takes bytes; this file is what produces them.

``torch`` is therefore an optional dependency (``pip install radfiled3d-nn[torch]``) and is imported
lazily — importing :mod:`RadFiled3D.nn` never imports it.

    from RadFiled3D.nn import deploy
    from RadFiled3D.nn.torch_export import export_to_onnx

    pkg = rfnn.PackageBuilder(dataset="xray-scatter-v3", software="radfield3d-nn 2.1")
    pkg.set_field_dimensions([1.0, 1.0, 1.0])
    pkg.add_input("position", "position", [3], unit="m")
    pkg.add_input("beam_direction", "beam_direction", [2], unit="rad")
    pkg.add_output("flux", "flux", [1])

    pkg.add_graph("trunk", export_to_onnx(
        model, (position, beam_direction),
        input_names=["position", "beam_direction"],
        output_names=["flux"],
    ))
    pkg.write("model.rf3m")

A model that factors into a beam encoder and a position trunk is two calls, one per graph — the
package carries both under their conventional names, ``beam_encoder`` and ``trunk``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Sequence

__all__ = ["batch_dim", "export_to_onnx", "graph_io_names", "add_graph", "verify_declarations"]


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as err:  # pragma: no cover - depends on the environment
        raise ModuleNotFoundError(
            "exporting a PyTorch module needs torch: pip install 'radfiled3d-nn[torch]'"
        ) from err
    return torch


def batch_dim() -> Any:
    """A dynamic batch dimension with an **explicit** upper bound.

    A stored model must accept any batch size at inference; without a dynamic axis the exporter
    freezes the traced example batch into the graph and the deployed model rejects every other size.

    The bound is not cosmetic. An unbounded ``Dim`` declares ``[0, int64_max]``, and some traced ops
    emit the guard ``batch != int64_max`` (the broadcast/expand sentinel), which that range cannot
    satisfy — the export then aborts with "Constraints violated (batch)". Bounding the dim to a
    realistic maximum removes the conflict without freezing the batch size.
    """
    torch = _torch()
    return torch.export.Dim("batch", min=1, max=2**31 - 1)


def export_to_onnx(
    module: Any,
    args: Sequence[Any],
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_batch: bool = True,
    dynamic_shapes: Any = None,
    **export_kwargs: Any,
) -> bytes:
    """Trace ``module`` and return the ONNX graph as bytes.

    ``input_names`` is applied **positionally** to the arguments that produce a graph input. A
    ``None`` argument produces no input, so listing a name for it shifts every later name onto the
    wrong tensor — which is silent, and fatal at deployment, because the runtime binds by name.
    Passing a ``None`` in ``args`` is therefore rejected rather than quietly renumbered.

    Args:
        module: the module to trace. Trace the *core* model, not a training wrapper.
        args: example inputs, positionally matching ``input_names``.
        input_names / output_names: the names the deployment runtime will bind. They must match the
            tensor names declared on the package, or the runtime cannot find them.
        dynamic_batch: give every input a dynamic batch axis (see :func:`batch_dim`).
        dynamic_shapes: pass your own instead, when a model needs more than a batch axis.
        **export_kwargs: forwarded to ``torch.onnx.export``.
    """
    torch = _torch()

    args = tuple(args)
    if any(a is None for a in args):
        raise ValueError(
            "a None argument produces no graph input, and input_names is applied positionally, so "
            "every later name would land on the wrong tensor. Drop the argument and its name."
        )
    if len(args) != len(input_names):
        raise ValueError(
            f"{len(args)} arguments but {len(input_names)} input names; they are matched positionally"
        )

    if dynamic_shapes is None and dynamic_batch:
        batch = batch_dim()
        dynamic_shapes = tuple({0: batch} for _ in args)

    program = torch.onnx.export(
        model=module,
        args=args,
        input_names=list(input_names),
        output_names=list(output_names),
        dynamic_shapes=dynamic_shapes,
        dynamo=True,
        **export_kwargs,
    )

    # `torch.onnx.export(dynamo=True)` returns an ONNXProgram that saves to a path. Going through a
    # temporary file rather than a buffer keeps this working across the versions where `save` takes
    # only a path; the graph is read straight back and never left on disk.
    with tempfile.TemporaryDirectory(prefix="rf3m-onnx-") as tmp:
        path = Path(tmp) / "graph.onnx"
        program.save(str(path))
        return path.read_bytes()


def graph_io_names(onnx_bytes: bytes) -> tuple[list[str], list[str]] | None:
    """The graph's declared input and output names, or ``None`` if ``onnx`` is not installed.

    Used to check an export against what the package declares. It is a separate, optional step
    because the check needs the ``onnx`` package, and a missing checker must not stop an export.
    """
    try:
        import onnx
    except ModuleNotFoundError:
        return None
    model = onnx.load_from_string(onnx_bytes)
    return (
        [i.name for i in model.graph.input],
        [o.name for o in model.graph.output],
    )


def add_graph(
    builder: Any,
    name: str,
    module: Any,
    args: Sequence[Any],
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    **export_kwargs: Any,
) -> bytes:
    """Export ``module`` and attach it to ``builder`` as the graph called ``name``.

    Returns the ONNX bytes, so a caller can inspect them or pass them to
    :func:`verify_declarations`.
    """
    onnx_bytes = export_to_onnx(
        module,
        args,
        input_names=input_names,
        output_names=output_names,
        **export_kwargs,
    )
    builder.add_graph(name, onnx_bytes)
    return onnx_bytes


def verify_declarations(builder: Any, *onnx_graphs: bytes) -> None:
    """Check that every tensor the package declares appears in one of its graphs.

    This is the mismatch worth guarding against: the package says it takes ``beam_dir`` while the
    graph declares ``beam_direction``. The deployment runtime binds **by name**, so the package
    loads and then cannot be driven — a failure that only shows up in the field.

    Both halves have to be checked together rather than per graph, because a model that factors into
    a beam encoder and a trunk splits its inputs across the two.

    Needs the ``onnx`` package; without it there is nothing to compare against and the call is a
    no-op rather than a false pass or a hard failure.
    """
    declared_inputs = set()
    declared_outputs = set()
    for tensor in builder.get_declared_tensors():
        (declared_inputs if tensor["role"] == "input" else declared_outputs).add(tensor["name"])

    graph_inputs: set[str] = set()
    graph_outputs: set[str] = set()
    for graph in onnx_graphs:
        names = graph_io_names(graph)
        if names is None:
            return
        graph_inputs.update(names[0])
        graph_outputs.update(names[1])

    missing_inputs = sorted(declared_inputs - graph_inputs)
    missing_outputs = sorted(declared_outputs - graph_outputs)
    if missing_inputs or missing_outputs:
        raise ValueError(
            "the package declares tensors no graph provides — the runtime binds by name, so this "
            f"package would load and then fail to bind. Missing inputs: {missing_inputs}, "
            f"missing outputs: {missing_outputs}. "
            f"Graph inputs: {sorted(graph_inputs)}, graph outputs: {sorted(graph_outputs)}"
        )
