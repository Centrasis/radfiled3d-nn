"""Describing a deployment package from Python.

The flow this covers is the one a trainer performs at the end of a run: state the header, declare
the dataset ranges and the inputs and outputs with their semantics and types, record the test
metrics, attach the ONNX graphs, and write. It must be expressible without reaching for the C++ API,
and everything written must come back out of `read_metadata` — a field the writer can set and the
reader cannot show is a field nobody can check.
"""

import pytest

from RadFiled3D.nn import deploy


def describe():
    """A package shaped like a real PBRFNet export, minus the actual graphs."""
    builder = deploy.PackageBuilder(
        dataset="DS03-Alderson-C-Arm",
        software="radfield3d-nn 2.1",
        physics="QGSP_BIC_HP+StdPhysics_Option4",
        # Supplied, never read from the clock, so building the same model twice gives the same bytes.
        created="2026-09-16T12:00:00Z",
    )
    builder.set_field_dimensions([1.0, 1.0, 1.0])
    builder.add_input("position", "position", [3], unit="m", range=(0.0, 1.0),
                      normalizer="linear0_1", normalizer_params=[0.0, 1.0])
    builder.add_input("tube_spectrum", "tube_spectrum", [150], unit="eV",
                      range=(0.0, 150_000.0, 1000.0))
    builder.add_input("source_distance", "source_distance", [1], unit="m", range=(0.5, 3.0))
    builder.add_output("flux", "flux", [1], normalizer="log_scale", normalizer_params=[1e-12, 30.0])
    builder.add_output("spectrum", "spectrum", [150], unit="eV")
    builder.add_graph("beam_encoder", b"<encoder onnx>")
    builder.add_graph("trunk", b"<trunk onnx>")
    builder.add_metric("test_loss", 0.4267)
    builder.add_metric("test_spectrum_accuracy", 0.8724)
    return builder


@pytest.fixture
def written(tmp_path):
    path = tmp_path / "model.rf3m"
    describe().write(str(path))
    return deploy.read_metadata(str(path))


def tensor(metadata, name):
    for t in metadata["tensors"]:
        if t["name"] == name:
            return t
    raise AssertionError(f"no tensor named {name}")


# ── the header ───────────────────────────────────────────────────────────────────────────────────


def test_the_header_round_trips(written):
    assert written["dataset"] == "DS03-Alderson-C-Arm"
    assert written["software"] == "radfield3d-nn 2.1"
    assert written["physics"] == "QGSP_BIC_HP+StdPhysics_Option4"
    assert written["created"] == "2026-09-16T12:00:00Z"


def test_a_package_built_twice_is_byte_identical():
    # `created` is a parameter rather than a clock read precisely so this holds.
    assert describe().to_bytes() == describe().to_bytes()


def test_an_undescribed_package_is_anonymous_not_rejected(tmp_path):
    # Worth pinning: the header is NOT required, so a caller who omits it gets empty strings rather
    # than an error. That is the format's choice, and a producer should know it is making it.
    path = tmp_path / "anon.rf3m"
    anon = deploy.PackageBuilder()
    anon.set_field_dimensions([1.0, 1.0, 1.0])
    anon.add_input("position", "position", [3], unit="m")
    anon.add_output("flux", "flux", [1])
    anon.add_graph("trunk", b"<trunk>")
    anon.write(str(path))
    assert deploy.read_metadata(str(path))["dataset"] == ""


# ── the interface: semantics, shapes, types ──────────────────────────────────────────────────────


def test_inputs_and_outputs_keep_their_semantics_and_shapes(written):
    assert tensor(written, "position")["semantic"] == "position"
    assert tensor(written, "position")["role"] == "input"
    assert tensor(written, "flux")["role"] == "output"
    # Resolution lives in the shape, and nowhere else.
    assert tensor(written, "spectrum")["shape"] == [150]


def test_a_semantic_this_build_never_heard_of_is_an_ordinary_call(tmp_path):
    # Rule 4: a new output must cost no format change. It round-trips and is reported as unknown
    # rather than refused.
    path = tmp_path / "novel.rf3m"
    builder = describe()
    builder.add_output("direction_distribution", "direction_distribution", [16, 32], unit="")
    builder.write(str(path))
    novel = tensor(deploy.read_metadata(str(path)), "direction_distribution")
    assert novel["shape"] == [16, 32]
    assert novel["known_semantic"] is False
    assert novel["semantic"] == "direction_distribution"


@pytest.mark.parametrize("dtype", ["f32", "f16", "i32", "u8"])
def test_every_element_type_round_trips(tmp_path, dtype):
    path = tmp_path / f"{dtype}.rf3m"
    builder = describe()
    builder.add_input("mask", "geometry_map", [64], dtype=dtype)
    builder.write(str(path))
    assert tensor(deploy.read_metadata(str(path)), "mask")["dtype"] == dtype


def test_an_unknown_element_type_is_refused():
    # Unlike a semantic, the dtype set is CLOSED: one the runtime cannot bind is a hard failure.
    with pytest.raises(ValueError, match="dtype"):
        describe().add_input("mask", "geometry_map", [64], dtype="bfloat16")


# ── the dataset ranges ───────────────────────────────────────────────────────────────────────────


def test_an_interval_range_round_trips(written):
    assert tensor(written, "source_distance")["range_kind"] == "min_max"
    assert tensor(written, "source_distance")["range"] == (0.5, 3.0)


def test_a_histogram_axis_round_trips(written):
    # What a tube spectrum needs: the bin width is what reconstructs the energy axis.
    assert tensor(written, "tube_spectrum")["range_kind"] == "histogram"
    assert tensor(written, "tube_spectrum")["range"] == (0.0, 150_000.0, 1000.0)


def test_category_labels_round_trip(tmp_path):
    path = tmp_path / "cat.rf3m"
    builder = describe()
    builder.add_input("collimator", "beam_collimation", [1], range=["rect", "circle"])
    builder.write(str(path))
    entry = tensor(deploy.read_metadata(str(path)), "collimator")
    assert entry["range_kind"] == "categorical"
    assert entry["range"] == ["rect", "circle"]


def test_a_tensor_with_no_range_says_so(written):
    assert tensor(written, "flux")["range_kind"] == "none"
    assert tensor(written, "flux")["range"] is None


def test_a_malformed_range_is_refused():
    with pytest.raises(ValueError, match="range"):
        describe().add_input("bad", "position", [3], range=(1.0, 2.0, 3.0, 4.0))


# ── normalizers, metrics, graphs ─────────────────────────────────────────────────────────────────


def test_normalizers_round_trip_with_their_parameters(written):
    assert tensor(written, "flux")["normalizer"] == "log_scale"
    assert tensor(written, "flux")["normalizer_params"] == [1e-12, 30.0]
    assert tensor(written, "position")["normalizer"] == "linear0_1"


def test_metrics_round_trip(written):
    assert written["metrics"]["test_loss"] == pytest.approx(0.4267)
    assert written["metrics"]["test_spectrum_accuracy"] == pytest.approx(0.8724)


def test_several_graphs_travel_in_one_package(written):
    # A model that factors into stages ships them all; each is a block with its own name.
    names = [b["name"] for b in written["blocks"]]
    assert set(names) == {"beam_encoder", "trunk"}
    assert all(b["kind"] == "onnx" for b in written["blocks"])


def test_a_package_without_a_trunk_is_refused(tmp_path):
    # The portable graph must always exist: a package whose only executable form is an accelerator
    # blob could not run anywhere else.
    builder = deploy.PackageBuilder(dataset="d", software="s", physics="p")
    builder.set_field_dimensions([1.0, 1.0, 1.0])
    builder.add_input("position", "position", [3], unit="m")
    builder.add_output("flux", "flux", [1])
    with pytest.raises(ValueError):
        builder.write(str(tmp_path / "no_trunk.rf3m"))


# ── the two kinds of model ───────────────────────────────────────────────────────────────────────


def test_a_voxelwise_model_may_not_fix_a_voxelization(tmp_path):
    # A model queried per position leaves the grid to the caller, so recording one would make it
    # silently wrong at every other grid.
    builder = describe()
    builder.set_voxelization([64, 64, 64], [0.015625, 0.015625, 0.015625])
    with pytest.raises(ValueError, match="voxel"):
        builder.write(str(tmp_path / "contradiction.rf3m"))


def test_a_whole_volume_model_records_its_grid(tmp_path):
    # No position input: a CNN emits the field in one run, and the grid is the model's own.
    path = tmp_path / "cnn.rf3m"
    builder = deploy.PackageBuilder(dataset="d", software="s", physics="p")
    builder.set_field_dimensions([1.0, 1.0, 1.0])
    builder.set_voxelization([64, 64, 64], [0.015625, 0.015625, 0.015625])
    builder.add_input("beam_direction", "beam_direction", [3])
    builder.add_output("flux", "flux", [1])
    builder.add_graph("trunk", b"<cnn onnx>")
    builder.write(str(path))
    assert deploy.read_metadata(str(path))["voxelization"]["voxel_counts"] == [64, 64, 64]


# ── graph composition ────────────────────────────────────────────────────────────────────────────


def composed(tmp_path):
    path = tmp_path / "composed.rf3m"
    builder = describe()
    # The memory between the stages. `elements` is per ROW; how many rows it holds is derived from
    # the shapes the graphs declare, so a latent computed once and read at every voxel is repeated.
    builder.add_buffer("beam_latent", 192)
    # The tensor names deliberately do not match: the exporter that wrote the packages on disk called
    # the encoder's output `linear_5` while the trunk's input is `latent`. The named buffer is what
    # reconciles them, and recording it is what lets a consumer run such a model at all.
    builder.add_stage("beam_encoder", "once", writes={"linear_5": "beam_latent"})
    builder.add_stage("trunk", "per_query", reads={"latent": "beam_latent"})
    builder.write(str(path))
    return deploy.read_metadata(str(path))


def test_the_wiring_round_trips(tmp_path):
    wiring = composed(tmp_path)["composition"]
    assert [s["name"] for s in wiring["stages"]] == ["beam_encoder", "trunk"]
    assert [s["invocation"] for s in wiring["stages"]] == ["once", "per_query"]
    assert wiring["buffers"] == [{"name": "beam_latent", "elements": 192, "dtype": "f32"}]
    assert wiring["stages"][0]["writes"] == {"linear_5": "beam_latent"}
    assert wiring["stages"][1]["reads"] == {"latent": "beam_latent"}
    # A tensor attached to no buffer is the caller's to bind.
    assert wiring["stages"][1]["writes"] == {}


def test_the_wiring_travels_as_a_block(tmp_path):
    # A block, not a metadata field, so rule 5 covers it: a tool that does not understand the
    # wiring writes it back verbatim instead of silently dropping it.
    blocks = {b["name"]: b["kind"] for b in composed(tmp_path)["blocks"]}
    assert blocks["composition"] == "composition"
    assert blocks["trunk"] == "onnx"


def test_a_package_without_wiring_says_so(written):
    # Absent is a meaning — a single trunk, driven directly — not a missing field.
    assert written["composition"] is None


def test_the_trunk_must_run_last(tmp_path):
    builder = describe()
    builder.add_stage("trunk", "per_query")
    builder.add_stage("beam_encoder", "once")
    with pytest.raises(ValueError, match="trunk"):
        builder.write(str(tmp_path / "wrong_order.rf3m"))


def test_a_stage_naming_no_block_is_refused(tmp_path):
    builder = describe()
    builder.add_stage("nonexistent", "once")
    builder.add_stage("trunk", "per_query")
    with pytest.raises(ValueError, match="nonexistent"):
        builder.write(str(tmp_path / "missing_graph.rf3m"))


def test_a_stage_may_not_read_a_buffer_a_later_stage_writes(tmp_path):
    # A buffer holds a value only once its producer has run, so the order has to match the dataflow.
    builder = describe()
    builder.add_buffer("from_the_future", 4)
    builder.add_stage("beam_encoder", "once", reads={"something": "from_the_future"})
    builder.add_stage("trunk", "per_query", writes={"flux": "from_the_future"})
    with pytest.raises(ValueError, match="earlier"):
        builder.write(str(tmp_path / "backwards.rf3m"))


# ── specialised code per stage ───────────────────────────────────────────────────────────────────


def specialised(tmp_path, write=True):
    """A package whose `hashgrid` stage exists only as vendor code — no portable ONNX form."""
    builder = describe()
    builder.add_block("cuda_ptx", "hashgrid", b"<ptx>")
    builder.add_block("hsaco", "hashgrid", b"<amd>", target_arch="gfx1100")
    builder.add_buffer("encoded", 32)
    builder.add_stage("hashgrid", "once", writes={"features": "encoded"})
    builder.add_stage_launch("hashgrid", "cuda_ptx", entry_point="hashgrid_encode",
                             block_size=(128, 1, 1))
    builder.add_stage_launch("hashgrid", "hsaco", entry_point="hashgrid_encode_amd",
                             target_arch="gfx1100", block_size=(64, 1, 1))
    builder.add_stage("trunk", "per_query", reads={"features": "encoded"})
    if not write:
        return builder
    path = tmp_path / "specialised.rf3m"
    builder.write(str(path))
    return deploy.read_metadata(str(path))


def test_a_stage_may_ship_only_vendor_code(tmp_path):
    # Only one implementation is required, and it need not be the portable one.
    stages = {s["name"]: s for s in specialised(tmp_path)["composition"]["stages"]}
    launches = stages["hashgrid"]["launches"]
    assert set(launches) == {"cuda_ptx", "hsaco"}
    # Per implementation, not per stage: the entry symbols genuinely differ.
    assert launches["cuda_ptx"]["entry_point"] == "hashgrid_encode"
    assert launches["cuda_ptx"]["block_size"] == [128, 1, 1]
    assert launches["hsaco"]["entry_point"] == "hashgrid_encode_amd"
    assert launches["hsaco"]["target_arch"] == "gfx1100"
    # A zero grid is derived from the work at run time, so one descriptor serves every resolution.
    assert launches["cuda_ptx"]["grid_size"] == [0, 0, 0]


def test_the_variants_are_blocks_not_a_list_in_the_stage(tmp_path):
    # Which machine code a stage has is already in the blocks; listing it in the composition too
    # would be a second source of truth about what the package holds.
    blocks = [(b["kind"], b["name"]) for b in specialised(tmp_path)["blocks"]]
    assert ("cuda_ptx", "hashgrid") in blocks
    assert ("hsaco", "hashgrid") in blocks
    assert ("onnx", "hashgrid") not in blocks


def test_a_launch_descriptor_may_not_describe_a_graph(tmp_path):
    builder = specialised(tmp_path, write=False)
    # ONNX Runtime is asked for a graph's calling convention by name; a second answer recorded
    # beside it could only ever disagree.
    builder.add_stage_launch("trunk", "onnx", entry_point="forward")
    with pytest.raises(ValueError, match="calling convention"):
        builder.write(str(tmp_path / "graph_launch.rf3m"))


def test_an_unknown_invocation_is_refused():
    with pytest.raises(ValueError, match="invocation"):
        describe().add_stage("trunk", "sometimes")


# ── stage weights ────────────────────────────────────────────────────────────────────────────────


def test_a_stages_weights_are_one_buffer_shared_by_its_implementations(tmp_path):
    builder = describe()
    builder.add_block("cuda_ptx", "trunk", b"<ptx>")
    builder.add_block("hsaco", "trunk", b"<amd>")
    builder.add_stage("trunk", "per_query")
    # ONE buffer, whichever implementation asks for it. A copy per implementation would be three
    # things that can drift apart.
    builder.add_stage_weights("trunk", b"<half-precision-parameters>", elements=13, dtype="f16",
                              onnx_external_file="trunk.onnx.data")
    path = tmp_path / "weighted.rf3m"
    builder.write(str(path))

    metadata = deploy.read_metadata(str(path))
    weights = metadata["composition"]["stages"][0]["weights"]
    assert weights == {"elements": 13, "dtype": "f16", "onnx_external_file": "trunk.onnx.data"}
    # Stored as a block, so a scan skips it by its length like any other payload — and it is not a
    # way to RUN the model.
    assert ("weights", "trunk") in [(b["kind"], b["name"]) for b in metadata["blocks"]]


def test_a_stage_with_no_weights_still_declares_the_buffer(tmp_path):
    # Empty, not missing: an implementation reads its weights the same way whether or not it has any.
    wiring = composed(tmp_path)["composition"]
    assert wiring["stages"][0]["weights"]["elements"] == 0
    assert "weights" not in [b["kind"] for b in composed(tmp_path)["blocks"]]


def test_weights_a_stage_declares_but_the_package_lacks_are_refused(tmp_path):
    # `add_stage_weights` stores the block and the declaration together, so this state is only
    # reachable by writing the composition by hand — but the reader must still refuse it.
    builder = describe()
    builder.add_stage("trunk", "per_query")
    builder.add_stage_weights("trunk", b"", elements=7)
    with pytest.raises(ValueError, match="weights"):
        builder.write(str(tmp_path / "phantom_weights.rf3m"))
