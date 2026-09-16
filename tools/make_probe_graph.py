#!/usr/bin/env python3
"""Regenerate `tests/probe_graph.hpp` — the tiny ONNX graph the field tests run.

The graph is embedded in the test suite as bytes rather than built at test time, because the C++
tests must run on a machine with no Python, no torch and no network. This script is the only place
its structure is defined; running it rewrites the header.

    python tools/make_probe_graph.py          # needs onnx; onnxruntime additionally verifies it

The graph is deliberately tiny and its output is derivable by hand, so a test can assert the exact
numbers a voxel grid must produce and therefore catch a wrong voxel ORDER, which a smoke test that
only checks for finite values cannot.
"""

import pathlib
import sys

import onnx
from onnx import TensorProto, helper

HEADER = pathlib.Path(__file__).resolve().parent.parent / "tests" / "probe_graph.hpp"

# A symbolic batch dimension: the session resolves it to the field's voxel count, so one graph
# serves every grid size a test wants.
QUERIES = "queries"


def build() -> bytes:
    position = helper.make_tensor_value_info("position", TensorProto.FLOAT, [QUERIES, 3])
    direction = helper.make_tensor_value_info("beam_direction", TensorProto.FLOAT, [3])
    flux = helper.make_tensor_value_info("flux", TensorProto.FLOAT, [QUERIES, 1])
    spectrum = helper.make_tensor_value_info("spectrum", TensorProto.FLOAT, [QUERIES, 4])

    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    nodes = [
        # beam_direction is CONSUMED, not merely declared: ORT prunes an input no node reads, and a
        # pruned input cannot exercise the "the caller still owes this binding" path.
        helper.make_node("Mul", ["position", "beam_direction"], ["weighted"]),
        helper.make_node("ReduceSum", ["weighted", "axes"], ["flux"], keepdims=1),
        helper.make_node("Concat", ["position", "flux"], ["spectrum"], axis=1),
    ]
    graph = helper.make_graph(
        nodes, "field_probe", [position, direction], [flux, spectrum], [axes]
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        producer_name="radfiled3d-nn tests",
    )
    # Pinned: a newer onnx defaults to an IR version the vendored ONNX Runtime may not accept yet,
    # and the failure is at session creation in an unrelated test.
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model.SerializeToString()


# The whole-volume graph's grid. Fixed by the "architecture", exactly as a real CNN's is, and chosen
# to match `test_geometry()` in tests/field_inference.cpp.
VOLUME_NX, VOLUME_NY, VOLUME_NZ = 2, 3, 4
VOLUME_VOXELS = VOLUME_NX * VOLUME_NY * VOLUME_NZ


def build_volume() -> bytes:
    """A whole-volume model: no position input, the whole grid emitted in one run.

    The output is shaped `[1, 1, nz, ny, nx]` — the channel-first, Z-slowest layout a convolutional
    network produces. In C order that indexes as `z*ny*nx + y*nx + x`, which is exactly RadFiled3D's
    own flat voxel index, so binding a layer directly is correct for a SINGLE-channel output. It is
    not for a multi-channel one, and nothing in the container says which layout such an output uses —
    which is why the runtime refuses those rather than guessing.

    `flux = sum(beam_direction) * arange(voxels)`, so with `beam_direction = (1, 1, 1)` voxel `i`
    holds `3 * i`: the value names its own flat index, and a transposed axis order fails loudly.
    """
    direction = helper.make_tensor_value_info("beam_direction", TensorProto.FLOAT, [3])
    flux = helper.make_tensor_value_info(
        "flux", TensorProto.FLOAT, [1, 1, VOLUME_NZ, VOLUME_NY, VOLUME_NX]
    )

    ramp = helper.make_tensor(
        "ramp", TensorProto.FLOAT, [1, 1, VOLUME_NZ, VOLUME_NY, VOLUME_NX],
        [float(i) for i in range(VOLUME_VOXELS)],
    )
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
    nodes = [
        # The input is consumed so it cannot be pruned, and so a test can vary the output by
        # rebinding it rather than by rebuilding the graph.
        helper.make_node("ReduceSum", ["beam_direction", "axes"], ["gain"], keepdims=0),
        helper.make_node("Mul", ["ramp", "gain"], ["flux"]),
    ]
    graph = helper.make_graph(nodes, "volume_probe", [direction], [flux], [ramp, axes])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        producer_name="radfiled3d-nn tests",
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model.SerializeToString()


def verify_volume(blob: bytes) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        return
    session = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
    (flux,) = session.run(None, {"beam_direction": np.ones(3, np.float32)})
    assert flux.shape == (1, 1, VOLUME_NZ, VOLUME_NY, VOLUME_NX), flux.shape
    assert np.allclose(flux.ravel(), 3.0 * np.arange(VOLUME_VOXELS)), flux.ravel()
    print(f"verified volume: shape={flux.shape} flux.flat[:4]={flux.ravel()[:4]}")


def build_encoder() -> bytes:
    """A beam encoder: beam parameters in, a latent row out. Runs ONCE per field.

    `latent = [[s, 2s]]` where `s = sum(beam_direction)`. Shape `[1, 2]` — one row, whatever the
    grid — which is exactly the case the composition has to repeat across the query batch.
    """
    direction = helper.make_tensor_value_info("beam_direction", TensorProto.FLOAT, [3])
    latent = helper.make_tensor_value_info("latent", TensorProto.FLOAT, [1, 2])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
    weights = helper.make_tensor("weights", TensorProto.FLOAT, [1, 2], [1.0, 2.0])
    nodes = [
        helper.make_node("ReduceSum", ["beam_direction", "axes"], ["s"], keepdims=0),
        helper.make_node("Mul", ["weights", "s"], ["latent"]),
    ]
    graph = helper.make_graph(nodes, "encoder_probe", [direction], [latent], [axes, weights])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)],
                              producer_name="radfiled3d-nn tests")
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model.SerializeToString()


def build_composed_trunk() -> bytes:
    """The trunk of the composed pair: position per query, latent per query, flux out.

    `flux = (x + y + z) + latent[0] + latent[1]`. With `beam_direction = (1, 1, 1)` the encoder emits
    `[3, 6]`, so voxel `q` holds `x + y + z + 9` — and it holds that for EVERY voxel only if the
    single encoder row was repeated correctly, which is the thing under test.
    """
    position = helper.make_tensor_value_info("position", TensorProto.FLOAT, [QUERIES, 3])
    latent = helper.make_tensor_value_info("latent", TensorProto.FLOAT, [QUERIES, 2])
    flux = helper.make_tensor_value_info("flux", TensorProto.FLOAT, [QUERIES, 1])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    nodes = [
        helper.make_node("ReduceSum", ["position", "axes"], ["p"], keepdims=1),
        helper.make_node("ReduceSum", ["latent", "axes"], ["l"], keepdims=1),
        helper.make_node("Add", ["p", "l"], ["flux"]),
    ]
    graph = helper.make_graph(nodes, "composed_trunk_probe", [position, latent], [flux], [axes])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)],
                              producer_name="radfiled3d-nn tests")
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model.SerializeToString()


# The name the external-weights graph refers to its initializer file by. It is recorded in the
# package (`Weights::onnx_external_file`) and satisfied FROM MEMORY at load, so no such file ever
# has to exist on disk beside the `.rf3m`.
WEIGHTS_FILE = "probe_weights.bin"
# `flux = dot(position, scale)` with the scale kept OUTSIDE the graph. Chosen so a wrong or missing
# weights hand-off cannot pass: with the correct values voxel (x,y,z) holds x + 10y + 100z, and with
# zeros — what an unsatisfied external initializer would leave — it holds 0.
WEIGHTS_VALUES = (1.0, 10.0, 100.0)


def build_external_weights_graph() -> tuple[bytes, bytes]:
    """A graph whose one initializer lives in an external file, plus that file's bytes.

    This is the ONNX half of the stage-weights contract: an exporter that keeps parameters out of the
    graph records a file name, and the runtime has to satisfy it from the package rather than from
    disk. A graph with its initializer embedded could not tell a working hand-off from a no-op.
    """
    import struct

    position = helper.make_tensor_value_info("position", TensorProto.FLOAT, [QUERIES, 3])
    flux = helper.make_tensor_value_info("flux", TensorProto.FLOAT, [QUERIES, 1])

    payload = b"".join(struct.pack("<f", v) for v in WEIGHTS_VALUES)
    # Built directly rather than through `make_tensor`, which insists on data: the whole point of an
    # external initializer is that the tensor carries a LOCATION and no numbers at all.
    scale = TensorProto()
    scale.name = "scale"
    scale.data_type = TensorProto.FLOAT
    scale.dims.append(3)
    scale.data_location = TensorProto.EXTERNAL
    for key, value in (("location", WEIGHTS_FILE), ("offset", "0"), ("length", str(len(payload)))):
        entry = scale.external_data.add()
        entry.key, entry.value = key, value

    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    nodes = [
        helper.make_node("Mul", ["position", "scale"], ["weighted"]),
        helper.make_node("ReduceSum", ["weighted", "axes"], ["flux"], keepdims=1),
    ]
    graph = helper.make_graph(nodes, "external_weights_probe", [position], [flux], [scale, axes])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)],
                              producer_name="radfiled3d-nn tests")
    model.ir_version = 9
    return model.SerializeToString(), payload


def verify_external_weights(blob: bytes, payload: bytes) -> None:
    """Prove the graph is inert without its weights and correct with them."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        return
    positions = np.array([[0.25, 0.5, 0.75], [1.0, 0.0, 0.0]], np.float32)
    expected = positions @ np.array(WEIGHTS_VALUES, np.float32)

    options = ort.SessionOptions()
    # Same three arguments the C++ side passes: the name the graph refers to, the bytes, the
    # length. Nothing on disk.
    #
    # `held` is a NAMED variable on purpose. ORT keeps the POINTER and reads it when the session is
    # constructed, which is after this call returns — pass a temporary and the initializer is read
    # from freed memory, which does not raise and does not produce zeros. It produces plausible
    # garbage, which is the failure mode this whole test exists to catch.
    held = bytearray(payload)
    options.add_external_initializers_from_files_in_memory([WEIGHTS_FILE], [held], [len(held)])
    session = ort.InferenceSession(blob, options, providers=["CPUExecutionProvider"])
    (flux,) = session.run(None, {"position": positions})
    assert np.allclose(flux.ravel(), expected), (flux, expected)
    print(f"verified external weights: flux={flux.ravel()} (expected {expected})")


def verify_composed(encoder: bytes, trunk: bytes) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        return
    e = ort.InferenceSession(encoder, providers=["CPUExecutionProvider"])
    (lat,) = e.run(None, {"beam_direction": np.ones(3, np.float32)})
    assert lat.shape == (1, 2) and np.allclose(lat, [[3.0, 6.0]]), lat
    t = ort.InferenceSession(trunk, providers=["CPUExecutionProvider"])
    pos = np.array([[0.25, 0.5, 0.75], [1.0, 1.0, 1.0]], np.float32)
    (flux,) = t.run(None, {"position": pos, "latent": np.repeat(lat, 2, axis=0)})
    assert np.allclose(flux.ravel(), pos.sum(axis=1) + 9.0), flux
    print(f"verified composed: latent={lat.ravel()} flux={flux.ravel()}")


def verify(blob: bytes) -> None:
    """Run the graph so a regenerated header is known good before it is written."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime/numpy not available — skipping the run check", file=sys.stderr)
        return
    session = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
    positions = np.array([[0.25, 0.25, 0.25], [0.75, 0.25, 0.25]], np.float32)
    flux, spectrum = session.run(None, {"position": positions, "beam_direction": np.ones(3, np.float32)})
    assert np.allclose(flux.ravel(), [0.75, 1.25]), flux
    assert np.allclose(spectrum[:, :3], positions), spectrum
    print(f"verified: flux={flux.ravel()} spectrum[0]={spectrum[0]}")


def rows_of(blob: bytes) -> str:
    return "\n".join(
        "    " + " ".join(f"0x{b:02x}," for b in blob[i : i + 16]) for i in range(0, len(blob), 16)
    )


def render(blob: bytes, volume: bytes, encoder: bytes, trunk: bytes,
           weighted: bytes, payload: bytes) -> str:
    rows = rows_of(blob)
    return f'''// A tiny ONNX graph whose output is PREDICTABLE from its input, for testing the field runtime.
//
// GENERATED by tools/make_probe_graph.py — do not edit by hand. It is embedded rather than produced
// at test time because the tests must run on a machine with no Python, no torch and no network.
//
// The graph, opset 18:
//
//     weighted = position * beam_direction        // [queries, 3], beam_direction broadcasts
//     flux     = ReduceSum(weighted, axis=1)      // [queries, 1]
//     spectrum = Concat(position, flux, axis=1)   // [queries, 4]
//
// With `beam_direction = (1, 1, 1)` that is `flux[q] = x + y + z` and
// `spectrum[q] = (x, y, z, x + y + z)` for the voxel centre `q`. Every number the field ends up
// holding is therefore derivable by hand, which is what lets the round-trip test assert the VOXEL
// ORDER and not merely that something was written.
//
// `beam_direction` is multiplied in rather than left dangling on purpose: ORT prunes an input no
// node consumes, and a pruned input cannot exercise the "the caller still owes this binding" path.
#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace rfnn_test {{

/// The serialised ONNX model, {len(blob)} bytes.
inline const std::vector<std::uint8_t>& probe_graph() {{
    static const std::vector<std::uint8_t> bytes = {{
{rows}
    }};
    return bytes;
}}

// ── the whole-volume counterpart ─────────────────────────────────────────────────────────────────
//
// No position input; the whole grid comes out of one run, the way a convolutional network works.
// The output is `[1, 1, nz, ny, nx]` — channel-first, Z-slowest — which in C order indexes as
// `z*ny*nx + y*nx + x`, RadFiled3D's own flat voxel index.
//
//     flux = sum(beam_direction) * arange(voxels)
//
// so with `beam_direction = (1, 1, 1)` voxel `i` holds `3 * i`: every value names its own flat
// index, and a transposed axis order fails on the numbers.

/// The grid this model's architecture fixes. A caller's field must match it.
inline constexpr std::uint32_t kVolumeNx = {VOLUME_NX}, kVolumeNy = {VOLUME_NY}, kVolumeNz = {VOLUME_NZ};

/// The serialised whole-volume model, {len(volume)} bytes.
inline const std::vector<std::uint8_t>& volume_probe_graph() {{
    static const std::vector<std::uint8_t> bytes = {{
{rows_of(volume)}
    }};
    return bytes;
}}

// ── a composed pair ──────────────────────────────────────────────────────────────────────────────
//
// What a real model looks like: a beam encoder that runs ONCE per field feeding a trunk that runs
// at every voxel.
//
//     encoder(beam_direction[3])                 -> latent[1, 2] = [s, 2s],  s = sum(direction)
//     trunk(position[q, 3], latent[q, 2])        -> flux[q, 1]   = (x+y+z) + latent[0] + latent[1]
//
// With `beam_direction = (1, 1, 1)` the encoder emits `[3, 6]` and voxel `q` must hold
// `x + y + z + 9`. It holds that for every voxel only if the encoder's SINGLE row was repeated
// across the query batch — which is the part of composition a two-graph package exists to exercise.

/// The beam encoder, {len(encoder)} bytes. One row out, whatever the grid.
inline const std::vector<std::uint8_t>& encoder_probe_graph() {{
    static const std::vector<std::uint8_t> bytes = {{
{rows_of(encoder)}
    }};
    return bytes;
}}

/// The trunk that consumes it, {len(trunk)} bytes.
inline const std::vector<std::uint8_t>& composed_trunk_graph() {{
    static const std::vector<std::uint8_t> bytes = {{
{rows_of(trunk)}
    }};
    return bytes;
}}

// ── a stage whose weights live outside its graph ──────────────────────────────────────────────────
//
// The ONNX half of the stage-weights contract. This graph's only initializer is EXTERNAL: it records
// the file name `{WEIGHTS_FILE}` instead of carrying the numbers, and the runtime satisfies that
// from the package's `weights` block in memory — nothing ever sits on disk beside the `.rf3m`.
//
//     flux[q] = dot(position[q], {WEIGHTS_VALUES})
//
// so voxel (x, y, z) must hold x + 10y + 100z. A hand-off that silently did nothing would leave the
// initializer unset, which is why the weights are a SCALE and not an offset: wrong weights give
// wrong numbers rather than plausible ones.

/// The external-data file name the graph refers to its weights by.
inline constexpr std::string_view kExternalWeightsFile = "{WEIGHTS_FILE}";

/// The graph, {len(weighted)} bytes. Inert until its weights are supplied.
inline const std::vector<std::uint8_t>& external_weights_graph() {{
    static const std::vector<std::uint8_t> bytes = {{
{rows_of(weighted)}
    }};
    return bytes;
}}

/// The weights themselves, {len(payload)} bytes: three `f32` scales.
inline const std::vector<std::uint8_t>& external_weights_payload() {{
    static const std::vector<std::uint8_t> bytes = {{
{rows_of(payload)}
    }};
    return bytes;
}}

}}  // namespace rfnn_test
'''


if __name__ == "__main__":
    data = build()
    verify(data)
    volume = build_volume()
    verify_volume(volume)
    encoder, trunk = build_encoder(), build_composed_trunk()
    verify_composed(encoder, trunk)
    weighted, payload = build_external_weights_graph()
    verify_external_weights(weighted, payload)
    HEADER.write_text(render(data, volume, encoder, trunk, weighted, payload))
    print(f"wrote {HEADER} ({len(data)} + {len(volume)} + {len(encoder)} + {len(trunk)}"
          f" + {len(weighted)} bytes)")
