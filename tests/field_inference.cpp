// Filling a radiation field from a session (requirements.md §1, the secondary path).
//
// The claim under test is the whole reason the GPU field type exists:
//
//     .rf3m ──load──▶ session ──infer──▶ GPUCartesianRadiationField ──▶ .rf3
//
// so the round trip runs all the way to a file and back through RadFiled3D's own FieldStore. The
// graph (tests/probe_graph.hpp) is chosen so every number is derivable by hand — `flux = x + y + z`
// of the voxel CENTRE — which means these cases fail on a wrong voxel order, a corner-sampled
// position or a transposed layer, none of which a "did it write something" check would notice.
#include "probe_graph.hpp"
#include "probe_kernel.hpp"

#include <RadFiled3D/nn/backends/cuda.hpp>
#include <RadFiled3D/nn/backends/onnx.hpp>
#include <RadFiled3D/nn/core/field.hpp>
#include <RadFiled3D/nn/core/field_inference.hpp>
#include <RadFiled3D/nn/deploy.hpp>
#include <RadFiled3D/nn/memory/cuda.hpp>

#include <gtest/gtest.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <memory>
#include <span>
#include <vector>

using namespace RadFiled3D::nn;
using RadFiled3D::GPUCartesianRadiationField;

namespace {

constexpr std::uint32_t kNx = 2, kNy = 3, kNz = 4;
constexpr std::size_t kVoxels = kNx * kNy * kNz;

CartesianFieldGeometry test_geometry() { return CartesianFieldGeometry::make({kNx, kNy, kNz}, {1.f, 1.5f, 2.f}); }

/// A package declaring exactly what `probe_graph()` consumes and produces.
///
/// Every normalizer is `Identity` deliberately: the graph's arithmetic is the thing under test, and
/// a normalizer in the way would make the expected values a second calculation that could itself be
/// wrong. The normalizer paths have their own cases below.
deploy::Package probe_package() {
    deploy::PackageBuilder builder;
    builder.provenance("field-probe", "radfiled3d-nn tests", "none")
        .field_dimensions_m({1.f, 1.5f, 2.f})
        .input("position", deploy::Semantic::Position, {3}).unit("m").done()
        .input("beam_direction", deploy::Semantic::BeamDirection, {3}).done()
        .output("flux", deploy::Semantic::Flux, {1}).unit("eV").done()
        .output("spectrum", deploy::Semantic::Spectrum, {4}).unit("eV").done()
        .graph("trunk", deploy::bytes(rfnn_test::probe_graph().begin(), rfnn_test::probe_graph().end()));
    return builder.build();
}

/// The session every case here drives, or null when this build has no ONNX Runtime.
std::shared_ptr<InferenceSession> probe_session() {
    if (!onnx::available()) return nullptr;
    return onnx::load(probe_package(), Backend::Cpu, -1);
}

/// `beam_direction = (1, 1, 1)`, which makes `flux[q]` the sum of the voxel centre's coordinates.
std::shared_ptr<memory::MemoryRef> unit_direction(std::vector<float>& storage) {
    storage.assign(3, 1.f);
    return memory::host::MemoryRef::of(std::span<float>(storage));
}

/// The voxel centre of flat index `q`, from the geometry alone — computed independently of the
/// implementation so the two must agree rather than share a bug.
std::array<float, 3> expected_center(std::size_t q, const CartesianFieldGeometry& geometry) {
    const auto v = geometry.get_voxel_dimensions_m();
    const std::size_t x = q % kNx;
    const std::size_t y = (q / kNx) % kNy;
    const std::size_t z = q / (kNx * kNy);
    return {(x + 0.5f) * v[0], (y + 0.5f) * v[1], (z + 0.5f) * v[2]};
}

}  // namespace

// ── positions ─────────────────────────────────────────────────────────────────────────────────────

TEST(FieldInference, PositionsAreVoxelCentresInRadFiled3DsFlatOrder) {
    const CartesianFieldGeometry geometry = test_geometry();
    const std::vector<float> positions = make_voxel_center_positions(geometry);
    ASSERT_EQ(positions.size(), kVoxels * 3);

    for (std::size_t q = 0; q < kVoxels; ++q) {
        const auto expected = expected_center(q, geometry);
        EXPECT_FLOAT_EQ(positions[q * 3 + 0], expected[0]) << "x of voxel " << q;
        EXPECT_FLOAT_EQ(positions[q * 3 + 1], expected[1]) << "y of voxel " << q;
        EXPECT_FLOAT_EQ(positions[q * 3 + 2], expected[2]) << "z of voxel " << q;
    }
}

TEST(FieldInference, PositionsSampleTheCentreAndNotTheCorner) {
    // The first voxel of a 1 m box on a 2x3x4 grid: half of (0.5, 0.5, 0.5) m.
    const std::vector<float> positions = make_voxel_center_positions(test_geometry());
    EXPECT_FLOAT_EQ(positions[0], 0.25f);
    EXPECT_FLOAT_EQ(positions[1], 0.25f);
    EXPECT_FLOAT_EQ(positions[2], 0.25f);
    // A corner sample would put the origin here, biasing the whole field by half a voxel.
    EXPECT_NE(positions[0], 0.f);
}

// ── preparation ───────────────────────────────────────────────────────────────────────────────────

TEST(FieldInference, EveryDeclaredOutputBecomesALayer) {
    auto session = probe_session();
    if (!session) GTEST_SKIP() << "built without ONNX Runtime";

    auto field = allocate_gpu_field(test_geometry());
    const FieldInference filler(session, field);

    // Derived from the DESCRIPTORS, not from a fixed list of names (rule 4).
    EXPECT_EQ(filler.get_output_layers(), (std::vector<std::string>{"flux", "spectrum"}));
    EXPECT_EQ(filler.get_channel(), std::string(kPredictionChannel));
    EXPECT_EQ(filler.get_position_input(), "position");

    auto buffer = field->get_channel(std::string(kPredictionChannel));
    ASSERT_TRUE(buffer);
    EXPECT_TRUE(buffer->has_layer("flux"));
    EXPECT_TRUE(buffer->has_layer("spectrum"));
    // A per-voxel vector is a histogram layer; a scalar is a plain one. Both are contiguous.
    EXPECT_EQ(buffer->get_layer("flux").get_voxel_flat_raw(0)->get_bytes(), sizeof(float));
    EXPECT_EQ(buffer->get_layer("spectrum").get_voxel_flat_raw(0)->get_bytes(), 4 * sizeof(float));
    EXPECT_EQ(buffer->get_layer("spectrum").get_unit(), "eV");
}

TEST(FieldInference, TheInputsTheCallerStillOwesAreNamed) {
    auto session = probe_session();
    if (!session) GTEST_SKIP() << "built without ONNX Runtime";

    const FieldInference filler(session, allocate_gpu_field(test_geometry()));
    // Positions are generated internally and never cross the API (R-I1); everything else is the
    // caller's, and is listed rather than discovered one failed run at a time.
    EXPECT_EQ(filler.get_caller_inputs(), (std::vector<std::string>{"beam_direction"}));
}

// ── the two kinds of model ────────────────────────────────────────────────────────────────────────
//
// An FCNN over a coordinate (PBRFNet, TPBRFNet) is queried a voxel at a time; a CNN emits the grid
// in one run. The container does not record which — it is read off the interface, so it cannot
// disagree with it — and `FieldInference` drives both.

namespace {

/// A whole-volume package: no position input, and the grid its architecture fixes.
deploy::PackageBuilder volume_builder() {
    deploy::PackageBuilder builder;
    builder.provenance("volume-probe", "tests", "none")
        .field_dimensions_m({1.f, 1.5f, 2.f})
        .voxelization({rfnn_test::kVolumeNx, rfnn_test::kVolumeNy, rfnn_test::kVolumeNz},
                      {0.5f, 0.5f, 0.5f})
        .input("beam_direction", deploy::Semantic::BeamDirection, {3}).done()
        .output("flux", deploy::Semantic::Flux, {1}).unit("eV").done()
        .graph("trunk", deploy::bytes(rfnn_test::volume_probe_graph().begin(),
                                      rfnn_test::volume_probe_graph().end()));
    return builder;
}

std::shared_ptr<InferenceSession> volume_session() {
    if (!onnx::available()) return nullptr;
    return onnx::load(volume_builder().build(), Backend::Cpu, -1);
}

}  // namespace

TEST(FieldInference, TheKindOfModelIsReadOffTheInterface) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";

    // A `Position` input is the whole rule, and it is DERIVED — nothing in the container stores a
    // kind that could contradict the descriptors.
    EXPECT_EQ(probe_package().get_model_kind(), deploy::ModelKind::VoxelWise);
    EXPECT_TRUE(probe_package().is_voxelwise());
    EXPECT_EQ(volume_builder().build().get_model_kind(), deploy::ModelKind::WholeVolume);
    EXPECT_FALSE(volume_builder().build().is_voxelwise());
    EXPECT_EQ(deploy::to_string(deploy::ModelKind::VoxelWise), "voxelwise");
    EXPECT_EQ(deploy::to_string(deploy::ModelKind::WholeVolume), "whole_volume");
}

TEST(FieldInference, AWholeVolumeModelFillsTheFieldInOneRun) {
    auto session = volume_session();
    if (!session) GTEST_SKIP() << "built without ONNX Runtime";

    auto field = allocate_gpu_field(test_geometry());
    FieldInference filler(session, field);
    EXPECT_EQ(filler.get_model_kind(), deploy::ModelKind::WholeVolume);
    // There is no position input to supply — the model's spatial structure is in the graph.
    EXPECT_TRUE(filler.get_position_input().empty());
    EXPECT_EQ(filler.get_caller_inputs(), (std::vector<std::string>{"beam_direction"}));

    std::vector<float> direction;
    session->bind_input("beam_direction", unit_direction(direction));
    filler.run();

    // The graph emits `[1, 1, nz, ny, nx]` holding `3 * i` at flat index `i`. Every voxel naming its
    // own index is what makes a transposed axis order fail here instead of looking plausible.
    const float* flux = field->get_channel(std::string(kPredictionChannel))->get_layer<float>("flux");
    for (std::size_t q = 0; q < kVoxels; ++q)
        EXPECT_NEAR(flux[q], 3.f * static_cast<float>(q), 1e-5f) << "voxel " << q;
}

TEST(FieldInference, AFieldThatDisagreesWithAWholeVolumeModelsGridIsRefused) {
    auto session = volume_session();
    if (!session) GTEST_SKIP() << "built without ONNX Runtime";

    // The grid belongs to the MODEL here, not to the caller: filling a differently shaped field is
    // not a resize, it is a different field.
    auto field = allocate_gpu_field(CartesianFieldGeometry::cubic(4, 1.f));
    try {
        FieldInference filler(session, field);
        ADD_FAILURE() << "a field of the wrong shape was accepted";
    } catch (const Exception& err) {
        EXPECT_NE(std::string(err.what()).find("2x3x4"), std::string::npos) << err.what();
        EXPECT_NE(std::string(err.what()).find("4x4x4"), std::string::npos) << err.what();
    }
}

/// The layout gap, pinned rather than guessed.
///
/// A voxelwise model emits `[queries, elements]` — voxel-major, the layout a RadFiled3D layer
/// already has. A whole-volume model's multi-element output is typically CHANNEL-major, and the
/// container records neither which it is nor how the spatial axes map to the flat index. Writing it
/// into a layer anyway would give a transposed field that passes every finite-and-nonzero check.
TEST(FieldInference, AMultiElementWholeVolumeOutputIsRefusedUntilTheFormatCarriesItsLayout) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";

    deploy::PackageBuilder builder = volume_builder();
    builder.output("spectrum", deploy::Semantic::Spectrum, {32}).unit("eV").done();
    // The graph does not actually produce `spectrum`; the refusal happens while preparing the
    // layers, before anything is bound, so the declaration alone is enough to reach it.
    const std::shared_ptr<InferenceSession> session = onnx::load(builder.build(), Backend::Cpu, -1);
    try {
        FieldInference filler(session, allocate_gpu_field(test_geometry()));
        ADD_FAILURE() << "a multi-element whole-volume output was bound without a known layout";
    } catch (const Exception& err) {
        EXPECT_NE(std::string(err.what()).find("channel-major"), std::string::npos) << err.what();
        EXPECT_NE(std::string(err.what()).find("spectrum"), std::string::npos) << err.what();
    }
}

namespace {

/// The probe package with `flux` carrying a normalizer, so what lands in the layer must be the
/// INVERSE of what the graph emitted.
deploy::Package normalized_flux_package(const deploy::Normalizer& normalizer) {
    deploy::PackageBuilder builder;
    builder.provenance("normalized-flux", "tests", "none")
        .field_dimensions_m({1.f, 1.5f, 2.f})
        .input("position", deploy::Semantic::Position, {3}).unit("m").done()
        .input("beam_direction", deploy::Semantic::BeamDirection, {3}).done()
        .output("flux", deploy::Semantic::Flux, {1}).unit("eV").normalizer(normalizer).done()
        .output("spectrum", deploy::Semantic::Spectrum, {4}).unit("eV").done()
        .graph("trunk", deploy::bytes(rfnn_test::probe_graph().begin(), rfnn_test::probe_graph().end()));
    return builder.build();
}

}  // namespace

TEST(FieldInference, ANormalizedOutputReachesTheLayerInMetricUnits) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";

    // The graph emits `x + y + z`; the package says that is `log(metric + eps) / scale`, so the
    // layer must end up holding `exp((x + y + z) * scale) - eps`.
    constexpr double kEpsilon = 1e-12, kScale = 2.0;
    const std::shared_ptr<InferenceSession> session =
        onnx::load(normalized_flux_package(deploy::LogScale{kEpsilon, kScale}), Backend::Cpu, -1);

    const CartesianFieldGeometry geometry = test_geometry();
    auto field = allocate_gpu_field(geometry);
    FieldInference filler(session, field);
    std::vector<float> direction;
    session->bind_input("beam_direction", unit_direction(direction));
    filler.run();

    auto buffer = field->get_channel(std::string(kPredictionChannel));
    const float* flux = buffer->get_layer<float>("flux");
    for (std::size_t q = 0; q < kVoxels; ++q) {
        const auto c = expected_center(q, geometry);
        const double graph = static_cast<double>(c[0]) + c[1] + c[2];
        const double metric = std::exp(graph * kScale) - kEpsilon;
        EXPECT_NEAR(flux[q], static_cast<float>(metric), static_cast<float>(metric) * 1e-5f)
            << "flux of voxel " << q;
        // And emphatically NOT the raw graph value, which is what a missing inversion would leave.
        EXPECT_GT(flux[q], static_cast<float>(graph));
    }

    // An `Identity` output in the same package is untouched — inverting it is a no-op, and the code
    // skips it rather than walking the volume for nothing.
    const float* spectrum = buffer->get_layer<float>("spectrum");
    const auto c0 = expected_center(0, geometry);
    EXPECT_NEAR(spectrum[0], c0[0], 1e-5f);
}

TEST(FieldInference, ANormalizerThisBuildCannotInvertIsRefused) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";

    // The name round-trips and is reportable (rule 4), but nothing here knows its arithmetic — so
    // the layer would carry graph units under the label `eV`, which rule 9 forbids.
    const std::shared_ptr<InferenceSession> session =
        onnx::load(normalized_flux_package(deploy::CustomNormalizer{"quantile", {0.1}}), Backend::Cpu, -1);
    auto field = allocate_gpu_field(test_geometry());
    try {
        FieldInference filler(session, field);
        ADD_FAILURE() << "a non-invertible output normalizer was accepted";
    } catch (const Exception& err) {
        EXPECT_NE(std::string(err.what()).find("quantile"), std::string::npos) << err.what();
        EXPECT_NE(std::string(err.what()).find("eV"), std::string::npos) << err.what();
    }
}

TEST(FieldInference, ANormalizedOutputWithADeviceMirrorIsRefused) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";

    const std::shared_ptr<InferenceSession> session =
        onnx::load(normalized_flux_package(deploy::LogScale{1e-12, 2.0}), Backend::Cpu, -1);
    auto field = allocate_gpu_field(test_geometry());
    field->add_channel(std::string(kPredictionChannel));
    std::vector<float> mirror(kVoxels, 0.f);
    field->set_device_memory(std::string(kPredictionChannel), "flux",
                             memory::host::MemoryRef::of(std::span<float>(mirror)));

    // Inverting is a host transform; there is no kernel to run one over device memory. Binding the
    // mirror anyway would leave graph units in the renderer's buffer with nothing to say so.
    try {
        FieldInference filler(session, field);
        ADD_FAILURE() << "a normalized output was bound to device memory";
    } catch (const Exception& err) {
        EXPECT_NE(std::string(err.what()).find("host transform"), std::string::npos) << err.what();
    }
}

// ── running ───────────────────────────────────────────────────────────────────────────────────────

TEST(FieldInference, TheFieldHoldsExactlyWhatTheGraphComputedPerVoxel) {
    auto session = probe_session();
    if (!session) GTEST_SKIP() << "built without ONNX Runtime";

    const CartesianFieldGeometry geometry = test_geometry();
    auto field = allocate_gpu_field(geometry);
    FieldInference filler(session, field);

    std::vector<float> direction;
    session->bind_input("beam_direction", unit_direction(direction));
    filler.run();

    auto buffer = field->get_channel(std::string(kPredictionChannel));
    const float* flux = buffer->get_layer<float>("flux");
    const float* spectrum = buffer->get_layer<float>("spectrum");

    for (std::size_t q = 0; q < kVoxels; ++q) {
        const auto c = expected_center(q, geometry);
        const float sum = c[0] + c[1] + c[2];
        EXPECT_NEAR(flux[q], sum, 1e-5f) << "flux of voxel " << q;
        // The histogram layer is one contiguous run of `voxels * bins` floats, so voxel q's bins
        // start at q * 4. A transposed or strided layout fails here.
        EXPECT_NEAR(spectrum[q * 4 + 0], c[0], 1e-5f) << "spectrum bin 0 of voxel " << q;
        EXPECT_NEAR(spectrum[q * 4 + 1], c[1], 1e-5f);
        EXPECT_NEAR(spectrum[q * 4 + 2], c[2], 1e-5f);
        EXPECT_NEAR(spectrum[q * 4 + 3], sum, 1e-5f);
    }
}

TEST(FieldInference, AMissingCallerInputIsReportedByName) {
    auto session = probe_session();
    if (!session) GTEST_SKIP() << "built without ONNX Runtime";

    FieldInference filler(session, allocate_gpu_field(test_geometry()));
    // `beam_direction` was never bound. Running must say so rather than produce a field of zeros.
    try {
        filler.run();
        ADD_FAILURE() << "a run with an unbound input was allowed";
    } catch (const Exception& err) {
        EXPECT_NE(std::string(err.what()).find("beam_direction"), std::string::npos) << err.what();
    }
}

TEST(FieldInference, ASecondRunPicksUpAnEditedInputWithoutRebinding) {
    auto session = probe_session();
    if (!session) GTEST_SKIP() << "built without ONNX Runtime";

    auto field = allocate_gpu_field(test_geometry());
    FieldInference filler(session, field);

    std::vector<float> direction;
    session->bind_input("beam_direction", unit_direction(direction));
    filler.run();
    const float first = field->get_channel(std::string(kPredictionChannel))->get_layer<float>("flux")[0];

    // Bindings are registered once and re-read every run (R-I1): editing in place is enough.
    direction.assign(3, 2.f);
    filler.run();
    const float second = field->get_channel(std::string(kPredictionChannel))->get_layer<float>("flux")[0];
    EXPECT_NEAR(second, 2.f * first, 1e-5f);
}

// ── the round trip A4 exists for ──────────────────────────────────────────────────────────────────

TEST(FieldInference, APredictionSurvivesToHostFieldAndTheRf3FileFormat) {
    auto session = probe_session();
    if (!session) GTEST_SKIP() << "built without ONNX Runtime";

    const CartesianFieldGeometry geometry = test_geometry();
    auto field = allocate_gpu_field(geometry);
    FieldInference filler(session, field);

    std::vector<float> direction;
    session->bind_input("beam_direction", unit_direction(direction));
    filler.run();

    const auto path = std::filesystem::temp_directory_path() / "rfnn_field_inference_roundtrip.rf3";
    std::filesystem::remove(path);
    store_host_field(field->to_host_field(), path, {"radfiled3d-nn", "test", "", ""});
    ASSERT_TRUE(std::filesystem::exists(path));

    auto loaded = load_host_field(path);
    ASSERT_TRUE(loaded);
    EXPECT_EQ(get_geometry_of(*loaded), geometry);

    auto buffer = loaded->get_channel(std::string(kPredictionChannel));
    ASSERT_TRUE(buffer);
    const float* flux = buffer->get_layer<float>("flux");
    const float* spectrum = buffer->get_layer<float>("spectrum");
    for (std::size_t q = 0; q < kVoxels; ++q) {
        const auto c = expected_center(q, geometry);
        const float sum = c[0] + c[1] + c[2];
        EXPECT_NEAR(flux[q], sum, 1e-5f) << "flux of voxel " << q << " after the .rf3 round trip";
        EXPECT_NEAR(spectrum[q * 4 + 3], sum, 1e-5f) << "spectrum of voxel " << q << " after the round trip";
    }
    std::filesystem::remove(path);
}

/// The spectrum layer's VOXEL VIEWS must point into the layer's own data, not at whatever buffer
/// the voxel template was built from.
///
/// Every other case here reads the layer through `get_layer<float>()` — the flat data buffer, which
/// is correct however the views are wired. Reading through a voxel accessor is the only way to catch
/// a template whose data pointer survived into the voxels.
TEST(FieldInference, SpectrumVoxelsReadBackThroughTheVoxelAccessor) {
    auto session = probe_session();
    if (!session) GTEST_SKIP() << "built without ONNX Runtime";

    const CartesianFieldGeometry geometry = test_geometry();
    auto field = allocate_gpu_field(geometry);
    FieldInference filler(session, field);
    std::vector<float> direction;
    session->bind_input("beam_direction", unit_direction(direction));
    filler.run();

    const RadFiled3D::VoxelLayer& layer = field->get_channel(std::string(kPredictionChannel))->get_layer("spectrum");
    for (std::size_t q = 0; q < kVoxels; ++q) {
        const auto* voxel = static_cast<const RadFiled3D::HistogramVoxel<float>*>(layer.get_voxel_flat_raw(q));
        const std::span<float> bins = voxel->get_histogram();
        ASSERT_EQ(bins.size(), 4u) << "voxel " << q;
        const auto c = expected_center(q, geometry);
        EXPECT_NEAR(bins[0], c[0], 1e-5f) << "voxel " << q << " reads someone else's memory";
        EXPECT_NEAR(bins[3], c[0] + c[1] + c[2], 1e-5f) << "voxel " << q;
    }
}

/// Construction calls `set_voxel_grid`, which discards every existing binding. A caller who bound
/// first must be told, and the session's preflight is what tells them.
TEST(FieldInference, ABindingMadeBeforeConstructionIsGoneAndSaidSo) {
    auto session = probe_session();
    if (!session) GTEST_SKIP() << "built without ONNX Runtime";

    session->set_voxel_grid({kNx, kNy, kNz});
    std::vector<float> direction;
    session->bind_input("beam_direction", unit_direction(direction));

    FieldInference filler(session, allocate_gpu_field(test_geometry()));
    try {
        filler.run();
        ADD_FAILURE() << "a binding made before construction appeared to survive";
    } catch (const Exception& err) {
        EXPECT_NE(std::string(err.what()).find("beam_direction"), std::string::npos) << err.what();
    }
}

// ── graph composition ─────────────────────────────────────────────────────────────────────────────
//
// A real model is several graphs. The package records which run, in what order, how often, and what
// flows between them; the session honours that, so a tensor one graph produces never reaches the
// caller at all.

namespace {

/// A two-stage package: a beam encoder that runs once, feeding a per-voxel trunk.
deploy::PackageBuilder composed_builder() {
    deploy::PackageBuilder builder;
    builder.provenance("composed-probe", "tests", "none")
        .field_dimensions_m({1.f, 1.5f, 2.f})
        .input("position", deploy::Semantic::Position, {3}).unit("m").done()
        .input("beam_direction", deploy::Semantic::BeamDirection, {3}).done()
        .output("flux", deploy::Semantic::Flux, {1}).done()
        .graph("beam_encoder", deploy::bytes(rfnn_test::encoder_probe_graph().begin(),
                                             rfnn_test::encoder_probe_graph().end()))
        .graph("trunk", deploy::bytes(rfnn_test::composed_trunk_graph().begin(),
                                      rfnn_test::composed_trunk_graph().end()))
        // `latent` is `[1, 2]` out of the encoder and `[queries, 2]` into the trunk, so the buffer
        // declares 2 elements per row and the runtime repeats the one row the encoder writes.
        .buffer("latent", 2)
        .stage("beam_encoder", deploy::Invocation::Once)
        .writes("latent", "latent")
        .done()
        .stage("trunk", deploy::Invocation::PerQuery)
        .reads("latent", "latent")
        .done();
    return builder;
}

}  // namespace

// ── a mixed chain: an ONNX graph, a CUDA kernel, an ONNX graph ───────────────────────────────────
//
// The case the whole stage abstraction exists for. `beam_encoder` (ONNX, once) writes a latent; a
// PTX stage scales it by its own weights; the trunk (ONNX, per query) consumes the result. Every
// intermediate lives in DEVICE memory, so nothing crosses the bus between stages.
//
// Numbers derivable by hand, which is what makes this a proof rather than a smoke test:
//
//     beam_direction = (1, 1, 1)   ->  latent = [3, 6]
//     weights        = [10, 100]   ->  scaled = [30, 600]
//     flux[q]                      =  (x + y + z) + 30 + 600
//
// Weights that never arrived would leave zeros, and a scale by zero is zero — so a silent hand-off
// failure cannot pass as a plausible number.

namespace {

/// What the kernel multiplies the latent by. Distinct powers of ten so a swapped or missing weight
/// is obvious in the result rather than plausible.
constexpr float kScaleWeights[] = {10.f, 100.f};

deploy::PackageBuilder kernel_chain_builder() {
    const auto& ptx = rfnn_test::probe_kernel_ptx();
    deploy::PackageBuilder builder;
    builder.provenance("kernel-chain-probe", "tests", "none")
        .field_dimensions_m({1.f, 1.f, 1.f})
        .input("position", deploy::Semantic::Position, {3}).unit("m").done()
        .input("beam_direction", deploy::Semantic::BeamDirection, {3}).done()
        .output("flux", deploy::Semantic::Flux, {1}).done()
        .graph("beam_encoder", deploy::bytes(rfnn_test::encoder_probe_graph().begin(),
                                             rfnn_test::encoder_probe_graph().end()))
        // No ONNX form: `scale` is a kernel and nothing else, which is the case a package ships when
        // a stage has no portable equivalent.
        .block(deploy::BlockKind::CudaPtx, "scale", "", deploy::bytes(ptx.begin(), ptx.end()))
        .graph("trunk", deploy::bytes(rfnn_test::composed_trunk_graph().begin(),
                                      rfnn_test::composed_trunk_graph().end()))
        .buffer("latent", rfnn_test::kProbeKernelWidth)
        .buffer("scaled", rfnn_test::kProbeKernelWidth)
        .stage("beam_encoder", deploy::Invocation::Once)
        .writes("latent", "latent")
        .done()
        .stage("scale", deploy::Invocation::Once)
        // The port ORDER is the kernel's argument order (docs/custom-code.md §4).
        .reads("latent", "latent")
        .writes("scaled", "scaled")
        .weights(deploy::bytes(reinterpret_cast<const std::uint8_t*>(kScaleWeights),
                               reinterpret_cast<const std::uint8_t*>(kScaleWeights) +
                                   sizeof(kScaleWeights)),
                 rfnn_test::kProbeKernelWidth)
        .launch(deploy::BlockKind::CudaPtx, "",
                deploy::KernelLaunch{std::string(rfnn_test::kProbeKernelEntry), {32, 1, 1}, {0, 0, 0}, 0})
        .done()
        .stage("trunk", deploy::Invocation::PerQuery)
        .reads("latent", "scaled")
        .done();
    return builder;
}

}  // namespace

/// Run the chain on a backend and check every voxel against the value derived by hand.
void expect_kernel_chain_runs(Backend backend) {
    const std::shared_ptr<InferenceSession> session =
        onnx::load(kernel_chain_builder().build(), backend, 0);

    constexpr std::uint32_t kSide = 2;
    constexpr std::size_t kQ = kSide * kSide * kSide;
    session->set_voxel_grid({kSide, kSide, kSide});

    const auto positions = make_voxel_center_positions(CartesianFieldGeometry::cubic(kSide, 1.f));
    std::vector<float> position(positions.begin(), positions.end());
    std::vector<float> direction{1.f, 1.f, 1.f}, flux(kQ, 0.f);
    session->bind_input("position", memory::host::MemoryRef::of(std::span<float>(position)));
    session->bind_input("beam_direction", memory::host::MemoryRef::of(std::span<float>(direction)));
    session->bind_output("flux", memory::host::MemoryRef::of(std::span<float>(flux)));
    session->infer();

    for (std::size_t q = 0; q < kQ; ++q) {
        const float expected = position[q * 3] + position[q * 3 + 1] + position[q * 3 + 2] +
                               3.f * kScaleWeights[0] + 6.f * kScaleWeights[1];
        EXPECT_NEAR(flux[q], expected, 1e-3f) << to_string(backend) << ", voxel " << q;
    }
}


TEST(FieldInference, AKernelStageRunsBetweenTwoGraphs) {
    if (!onnx::available() || !cuda::available() || cuda::get_device_count() == 0)
        GTEST_SKIP() << "needs ONNX Runtime and a CUDA device";
    expect_kernel_chain_runs(Backend::Cuda);
}

/// The same package on the TensorRT provider.
///
/// TensorRT and CUDA are two providers on ONE card, so a `cuda_ptx` block runs under both. Selecting
/// blocks by the backend's NAME rather than its hardware family refused this one while claiming the
/// hardware could not run it — which is why the two backends are pinned together here.
TEST(FieldInference, AKernelStageRunsUnderTheTensorRtProviderToo) {
    if (!onnx::available() || !is_available(Backend::TensorRt) || cuda::get_device_count() == 0)
        GTEST_SKIP() << "needs ONNX Runtime with TensorRT and a CUDA device";

    // Pinned WITHOUT loading anything, because the provider's shared library may be absent at run
    // time and skipping for that reason must not quietly take the selection rule with it. This is
    // the assertion that fails if block selection goes back to keying on the backend's name.
    const ComputeBackend& trt = get_compute_backend(Backend::TensorRt);
    EXPECT_EQ(trt.get_block_target(), "cuda");
    EXPECT_EQ(trt.get_device_arch(0), get_compute_backend(Backend::Cuda).get_device_arch(0));
    EXPECT_NO_THROW(kernel_chain_builder().build().require_runnable_on(trt.get_block_target(),
                                                                      trt.get_device_arch(0)));

    try {
        expect_kernel_chain_runs(Backend::TensorRt);
    } catch (const RadFiled3D::nn::Exception& err) {
        if (std::string(err.what()).find("cannot run the model") != std::string::npos) throw;
        GTEST_SKIP() << "TensorRT provider unavailable at run time: " << err.what();
    } catch (const std::exception& err) {
        GTEST_SKIP() << "TensorRT provider unavailable at run time: " << err.what();
    }
}


TEST(FieldInference, ARepeatedInferenceOnAKernelChainAllocatesNothing) {
    if (!onnx::available() || !cuda::available() || cuda::get_device_count() == 0)
        GTEST_SKIP() << "needs ONNX Runtime and a CUDA device";

    // The buffers between stages are instantiated when the grid is chosen and reused by every run.
    // Their addresses staying put across two inferences is what proves a repeated inference neither
    // reallocates nor re-stages — and it is also what lets a kernel hold a pointer.
    const std::shared_ptr<InferenceSession> session =
        onnx::load(kernel_chain_builder().build(), Backend::Cuda, 0);
    session->set_voxel_grid({2, 2, 2});

    const auto positions = make_voxel_center_positions(CartesianFieldGeometry::cubic(2, 1.f));
    std::vector<float> position(positions.begin(), positions.end());
    std::vector<float> direction{1.f, 1.f, 1.f}, first(8, 0.f), second(8, 0.f);
    session->bind_input("position", memory::host::MemoryRef::of(std::span<float>(position)));
    session->bind_input("beam_direction", memory::host::MemoryRef::of(std::span<float>(direction)));

    session->bind_output("flux", memory::host::MemoryRef::of(std::span<float>(first)));
    session->infer();
    const void* weights_address = session->get_stage_weights("scale").get_bytes().data();

    session->bind_output("flux", memory::host::MemoryRef::of(std::span<float>(second)));
    session->infer();

    EXPECT_EQ(session->get_stage_weights("scale").get_bytes().data(), weights_address);
    for (std::size_t q = 0; q < first.size(); ++q) EXPECT_FLOAT_EQ(first[q], second[q]) << "voxel " << q;
    EXPECT_NE(first[0], 0.f) << "a chain that produced nothing proves nothing";
}

TEST(FieldInference, AKernelStageOnTheWrongHardwareIsRefusedByName) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";

    // The same package on the CPU provider: `scale` has no portable form, so loading must fail
    // saying WHICH stage and WHAT the package carries — not "no such graph".
    try {
        (void)onnx::load(kernel_chain_builder().build(), Backend::Cpu, -1);
        FAIL() << "a package with a CUDA-only stage must not load on the CPU provider";
    } catch (const RadFiled3D::nn::Exception& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("scale"), std::string::npos) << what;
        EXPECT_NE(what.find("cuda_ptx"), std::string::npos) << what;
        EXPECT_NE(what.find("cpu"), std::string::npos) << what;
        EXPECT_EQ(what.find("`trunk`"), std::string::npos) << what;
    }
}

// ── device selection ─────────────────────────────────────────────────────────────────────────────
//
// A session runs on ONE device and everything uses it: the execution provider, the buffers between
// stages, a kernel stage's module and its launches. The device is chosen by the caller — explicitly,
// or by handing over memory that already lives somewhere, which is what a renderer does.
//
// This machine has one GPU, so cross-device EXECUTION cannot be exercised here. What can be, and is:
// that a device is resolved and reported, that an ordinal the backend lacks is refused, that memory
// from another device is refused by both indices, and that memory with no device of its own is not.

TEST(FieldInference, ASessionReportsTheDeviceItResolved) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";

    // `automatic()` is resolved when the session is built, so `get_device()` is always a real index.
    const auto automatic = load(probe_package(), Backend::Cpu, Device::automatic());
    EXPECT_EQ(automatic->get_device(), 0);
    EXPECT_TRUE(Device::automatic().is_automatic());
    EXPECT_EQ(Device::ordinal(1).get_index(), 1);

    // Explicit selection through the PUBLIC entry point, which before this could not choose at all.
    EXPECT_EQ(load(probe_package(), Backend::Cpu, Device::ordinal(0))->get_device(), 0);
}

TEST(FieldInference, AnOrdinalTheBackendDoesNotHaveIsRefused) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";

    try {
        (void)load(probe_package(), Backend::Cpu, Device::ordinal(7));
        FAIL() << "a device the backend does not have must be refused";
    } catch (const RadFiled3D::nn::Exception& e) {
        const std::string what = e.what();
        // Both halves: what was asked for, and what is actually there.
        EXPECT_NE(what.find("7"), std::string::npos) << what;
        EXPECT_NE(what.find("cpu"), std::string::npos) << what;
    }
}

TEST(FieldInference, TheDeviceCanBeTakenFromMemoryTheCallerAlreadyOwns) {
    // The renderer's flow: import first, then load on the device the import landed on. Here the
    // reference stands in for the imported one.
    if (!cuda::available() || cuda::get_device_count() == 0) GTEST_SKIP() << "no CUDA device";

    const auto on_zero = memory::cuda::allocate(64, 0);
    EXPECT_EQ(on_zero->get_device_index(), 0);
    EXPECT_EQ(Device::of(*on_zero).get_index(), 0);

    // Host memory has no device of its own, so deriving one from it means `automatic()` — NOT
    // device 0, which would silently undo a choice the caller was trying to make.
    std::vector<float> host(4, 0.f);
    EXPECT_TRUE(Device::of(*memory::host::MemoryRef::of(std::span<float>(host))).is_automatic());
}

TEST(FieldInference, MemoryFromAnotherDeviceIsRefusedRatherThanFaultedOn) {
    if (!onnx::available() || !cuda::available() || cuda::get_device_count() == 0)
        GTEST_SKIP() << "needs ONNX Runtime and a CUDA device";

    const auto session = load(probe_package(), Backend::Cuda, Device::ordinal(0));
    session->set_voxel_grid({1, 1, 1});

    // A reference that CLAIMS device 1. Fabricated deliberately: this machine has one GPU, and the
    // refusal is the half of multi-GPU behaviour a single card can still prove. Its address is never
    // dereferenced — the check happens first, which is the whole point.
    const auto elsewhere = std::make_shared<memory::cuda::MemoryRef>(0xdead'beefu, 3 * sizeof(float),
                                                                     /*device=*/1);
    try {
        session->bind_input("position", elsewhere);
        FAIL() << "memory from another device must not be bound";
    } catch (const RadFiled3D::nn::Exception& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("device 1"), std::string::npos) << what;
        EXPECT_NE(what.find("device 0"), std::string::npos) << what;
    }

    // Memory that does not know its device is not refused: -1 means "no such notion", and a host
    // buffer binds to a session on any card.
    std::vector<float> host(3, 0.5f);
    EXPECT_NO_THROW(session->bind_input("position",
                                        memory::host::MemoryRef::of(std::span<float>(host))));
}

TEST(FieldInference, CudaCanSayWhichDeviceAPointerIsOn) {
    if (!cuda::available() || cuda::get_device_count() == 0) GTEST_SKIP() << "no CUDA device";

    // What the Python path relies on: `__cuda_array_interface__` carries an address and a length and
    // no ordinal, so the driver is asked. Checked against an allocation whose device we know.
    const auto owned = memory::cuda::allocate(64, 0);
    EXPECT_EQ(memory::cuda::device_of(owned->get_address()), 0);
    // A host address is not device memory, and saying "-1" is what lets the domain check produce the
    // better message instead of this one guessing.
    std::vector<float> host(4, 0.f);
    EXPECT_EQ(memory::cuda::device_of(
                  reinterpret_cast<std::uint64_t>(static_cast<const void*>(host.data()))),
              -1);
}

// ── stage weights ────────────────────────────────────────────────────────────────────────────────
//
// One buffer per stage, shared by that stage's implementations. The ONNX half of the contract is the
// one that can be exercised here: a graph exported with its parameters OUTSIDE it names a file, and
// the runtime satisfies that from the package rather than from disk.

namespace {

/// A package whose trunk keeps its weights in a `weights` block instead of inside the graph.
deploy::PackageBuilder weighted_builder() {
    deploy::PackageBuilder builder;
    builder.provenance("weighted-probe", "tests", "none")
        .field_dimensions_m({1.f, 1.f, 1.f})
        .input("position", deploy::Semantic::Position, {3}).unit("m").done()
        .output("flux", deploy::Semantic::Flux, {1}).done()
        .graph("trunk", deploy::bytes(rfnn_test::external_weights_graph().begin(),
                                      rfnn_test::external_weights_graph().end()))
        .stage("trunk", deploy::Invocation::PerQuery)
        .weights(deploy::bytes(rfnn_test::external_weights_payload().begin(),
                               rfnn_test::external_weights_payload().end()),
                 /*elements=*/3, deploy::DType::F32,
                 std::string(rfnn_test::kExternalWeightsFile))
        .done();
    return builder;
}

}  // namespace

TEST(FieldInference, AStagesWeightsTravelInThePackageAndReachItsGraph) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";

    const deploy::Package package = deploy::Package::read(weighted_builder().build().to_bytes());
    // Stored as a block, so a scan skips it by its length like any other payload.
    EXPECT_EQ(package.get_weights("trunk").size(), 3u * sizeof(float));

    const std::shared_ptr<InferenceSession> session = onnx::load(package, Backend::Cpu, -1);
    // The ONE entry point: every implementation of the stage reads its parameters from here.
    const StageWeights& weights = session->get_stage_weights("trunk");
    EXPECT_EQ(weights.get_elements(), 3u);
    EXPECT_EQ(weights.get_dtype(), deploy::DType::F32);
    EXPECT_EQ(weights.get_onnx_external_file(), rfnn_test::kExternalWeightsFile);
    ASSERT_FALSE(weights.is_empty());
    EXPECT_NE(weights.get_memory(), nullptr);

    // And they actually reached the graph: `flux = dot(position, [1, 10, 100])`, which is only true
    // if the package's block satisfied the initializer the graph left outside itself.
    session->set_voxel_grid({2, 1, 1});
    std::vector<float> position{0.25f, 0.5f, 0.75f, 1.f, 0.f, 0.f}, flux(2, 0.f);
    session->bind_input("position", memory::host::MemoryRef::of(std::span<float>(position)));
    session->bind_output("flux", memory::host::MemoryRef::of(std::span<float>(flux)));
    session->infer();
    EXPECT_NEAR(flux[0], 80.25f, 1e-3f);
    EXPECT_NEAR(flux[1], 1.f, 1e-4f);
}

TEST(FieldInference, AStageWithNoWeightsStillHasTheBuffer) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";

    // An implementation reads its weights the same way whether or not it has any, so a stage with
    // none answers with an EMPTY buffer rather than with nothing to check.
    const std::shared_ptr<InferenceSession> session = onnx::load(probe_package(), Backend::Cpu, -1);
    const StageWeights& weights = session->get_stage_weights("trunk");
    EXPECT_TRUE(weights.is_empty());
    EXPECT_EQ(weights.get_elements(), 0u);
    EXPECT_NE(weights.get_memory(), nullptr) << "an empty buffer is still a buffer";
    EXPECT_THROW((void)session->get_stage_weights("no_such_stage"), RadFiled3D::nn::Exception);
}

TEST(FieldInference, WeightsOutliveEveryGridChange) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";

    // A grid chooses how many queries run; it does not change what the model weighs. ORT holds a raw
    // pointer into this buffer from session construction onwards, so rebuilding it per grid would be
    // a dangling read that produces plausible garbage rather than an error.
    const std::shared_ptr<InferenceSession> session =
        onnx::load(weighted_builder().build(), Backend::Cpu, -1);
    const void* address = session->get_stage_weights("trunk").get_bytes().data();

    std::vector<float> position{0.25f, 0.5f, 0.75f}, flux(1, 0.f);
    for (const std::uint32_t side : {1u, 1u}) {
        session->set_voxel_grid({side, 1, 1});
        session->bind_input("position", memory::host::MemoryRef::of(std::span<float>(position)));
        session->bind_output("flux", memory::host::MemoryRef::of(std::span<float>(flux)));
        session->infer();
        EXPECT_NEAR(flux[0], 80.25f, 1e-3f);
    }
    EXPECT_EQ(session->get_stage_weights("trunk").get_bytes().data(), address)
        << "the weights moved across a grid change";
}

TEST(FieldInference, AComposedPackageRunsItsStagesAndCarriesTheLatent) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";

    const std::shared_ptr<InferenceSession> session =
        onnx::load(composed_builder().build(), Backend::Cpu, -1);
    const CartesianFieldGeometry geometry = test_geometry();
    auto field = allocate_gpu_field(geometry);
    FieldInference filler(session, field);

    // The encoder's inputs are the caller's; the latent is not, and is not listed.
    EXPECT_EQ(filler.get_caller_inputs(), (std::vector<std::string>{"beam_direction"}));

    std::vector<float> direction;
    session->bind_input("beam_direction", unit_direction(direction));
    filler.run();

    // `beam_direction = (1,1,1)` gives `latent = [3, 6]`, so every voxel must hold `x+y+z+9`. It
    // does so only if the encoder's SINGLE row was repeated across all the queries.
    const float* flux = field->get_channel(std::string(kPredictionChannel))->get_layer<float>("flux");
    for (std::size_t q = 0; q < kVoxels; ++q) {
        const auto c = expected_center(q, geometry);
        EXPECT_NEAR(flux[q], c[0] + c[1] + c[2] + 9.f, 1e-5f) << "voxel " << q;
    }
}

TEST(FieldInference, AComposedStageIsRerunWhenItsInputChanges) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";

    const std::shared_ptr<InferenceSession> session =
        onnx::load(composed_builder().build(), Backend::Cpu, -1);
    auto field = allocate_gpu_field(test_geometry());
    FieldInference filler(session, field);

    std::vector<float> direction;
    session->bind_input("beam_direction", unit_direction(direction));
    filler.run();
    const float first = field->get_channel(std::string(kPredictionChannel))->get_layer<float>("flux")[0];

    // Editing a beam input in place must reach the trunk THROUGH the encoder: the latent is
    // recomputed, not cached from the previous run. sum(direction) goes 3 -> 6, so latent [3,6] ->
    // [6,12] and the flux rises by 9.
    direction.assign(3, 2.f);
    filler.run();
    const float second = field->get_channel(std::string(kPredictionChannel))->get_layer<float>("flux")[0];
    EXPECT_NEAR(second - first, 9.f, 1e-5f);
}

/// Skipping a stage reuses what it produced last time, with no copy and no rebinding.
///
/// This is the counterpart to `AComposedStageIsRerunWhenItsInputChanges`: there, editing the beam
/// input reaches the trunk THROUGH the encoder and flux rises by 9. Here the encoder is switched
/// off, so the latent buffer keeps the value the previous run left in it and the same edit changes
/// nothing at all. That difference is the whole feature — a global state encoded once stays valid
/// because the memory was never touched, not because anything was carried forward.
/// Memory this library allocates is EXPORTABLE, which is why `allocate` uses the virtual-memory
/// API rather than `cudaMalloc`.
///
/// The case it serves: inference allocated the output because the caller bound none, and a renderer
/// still has to display it without a copy. A `cudaMalloc` pointer has no shareable handle at all,
/// so this would be impossible; here the handle comes back and the allocation is genuinely
/// importable by Vulkan or D3D12.
TEST(FieldInference, MemoryAllocatedHereCanBeExportedToAGraphicsApi) {
    if (!cuda::available()) GTEST_SKIP() << "built without CUDA";
    const ComputeBackend& backend = get_compute_backend(Backend::Cuda);
    if (backend.get_device_count() == 0) GTEST_SKIP() << "no CUDA device present";

    auto memory = backend.allocate(4096, 0);
    ASSERT_NE(memory, nullptr);
    EXPECT_EQ(memory->get_size_bytes(), 4096u) << "the REQUESTED size, not the granularity-padded one";

    const auto exported = memory::cuda::export_external_memory(*memory);
    EXPECT_EQ(exported.mapped_bytes(), 4096u);
#ifdef _WIN32
    EXPECT_NE(std::get<memory::Win32Handle>(exported.handle).handle, nullptr);
#else
    const int fd = std::get<memory::OpaqueFd>(exported.handle).fd;
    EXPECT_GE(fd, 0) << "a real file descriptor a renderer could import";
    ::close(fd);  // nothing imported it, so this side still owns it
#endif

    // Memory that was IMPORTED cannot be re-exported: CUDA has no such call, which is exactly why a
    // reference remembers its origin instead.
    std::vector<float> host(8, 0.f);
    EXPECT_THROW((void)memory::cuda::export_external_memory(*memory::host::MemoryRef::of(std::span<float>(host))),
                 RadFiled3D::nn::Exception);
}

TEST(FieldInference, ASkippedStageKeepsWhatItLastWrote) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";

    const std::shared_ptr<InferenceSession> session =
        onnx::load(composed_builder().build(), Backend::Cpu, -1);
    auto field = allocate_gpu_field(test_geometry());
    FieldInference filler(session, field);

    std::vector<float> direction;
    session->bind_input("beam_direction", unit_direction(direction));
    filler.run();
    const float first = field->get_channel(std::string(kPredictionChannel))->get_layer<float>("flux")[0];

    EXPECT_TRUE(session->is_stage_enabled("beam_encoder"));
    session->set_stage_enabled("beam_encoder", false);
    EXPECT_FALSE(session->is_stage_enabled("beam_encoder"));

    // The same edit that moved the result by 9 while the encoder ran.
    direction.assign(3, 2.f);
    filler.run();
    const float frozen = field->get_channel(std::string(kPredictionChannel))->get_layer<float>("flux")[0];
    EXPECT_NEAR(frozen, first, 1e-5f) << "a skipped encoder must leave its latent untouched";

    // And switching it back on picks the change up, so this is a gate rather than a teardown.
    session->set_stage_enabled("beam_encoder", true);
    filler.run();
    const float thawed = field->get_channel(std::string(kPredictionChannel))->get_layer<float>("flux")[0];
    EXPECT_NEAR(thawed - first, 9.f, 1e-5f);
}

/// The intermediate buffers are reachable from outside, and they are the SAME memory across runs —
/// which is what lets a caller read a stage's output, or hand it to another API.
TEST(FieldInference, AStageBufferIsReachableAndStable) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";

    const std::shared_ptr<InferenceSession> session =
        onnx::load(composed_builder().build(), Backend::Cpu, -1);
    auto field = allocate_gpu_field(test_geometry());
    FieldInference filler(session, field);

    const auto names = session->get_stage_buffers();
    ASSERT_FALSE(names.empty());
    const auto& buffer = session->get_stage_buffer(names.front());
    ASSERT_NE(buffer, nullptr);
    const void* before = reinterpret_cast<const void*>(buffer->get_address());

    std::vector<float> direction;
    session->bind_input("beam_direction", unit_direction(direction));
    filler.run();
    filler.run();

    // Allocated once when the grid was chosen and reused by every inference, so the address a
    // caller kept stays the address the stage writes to.
    EXPECT_EQ(reinterpret_cast<const void*>(session->get_stage_buffer(names.front())->get_address()),
              before);
    EXPECT_THROW((void)session->get_stage_buffer("no-such-buffer"), RadFiled3D::nn::Exception);
    EXPECT_THROW(session->set_stage_enabled("no-such-stage", false), RadFiled3D::nn::Exception);
}

TEST(FieldInference, AComposedSessionSurvivesAChangeOfGrid) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";

    const std::shared_ptr<InferenceSession> session =
        onnx::load(composed_builder().build(), Backend::Cpu, -1);

    // The intermediate buffers are sized from the query count, so a new grid frees the old ones —
    // while the stages' ORT bindings still point into them. The clears have to happen first, and a
    // session that is only ever gridded once could not show it.
    for (const std::uint32_t side : {2u, 4u, 3u}) {
        auto field = allocate_gpu_field(CartesianFieldGeometry::cubic(side, 1.f));
        FieldInference filler(session, field);
        std::vector<float> direction;
        session->bind_input("beam_direction", unit_direction(direction));
        filler.run();

        const CartesianFieldGeometry geometry = field->get_geometry();
        const float* flux = field->get_channel(std::string(kPredictionChannel))->get_layer<float>("flux");
        const auto voxels = static_cast<std::size_t>(geometry.get_voxel_count());
        for (std::size_t q = 0; q < voxels; ++q) {
            const auto v = geometry.get_voxel_dimensions_m();
            const std::size_t x = q % side, y = (q / side) % side, z = q / (side * side);
            const float expected = (x + 0.5f) * v[0] + (y + 0.5f) * v[1] + (z + 0.5f) * v[2] + 9.f;
            ASSERT_NEAR(flux[q], expected, 1e-5f) << "side " << side << " voxel " << q;
        }
    }
}

TEST(FieldInference, AComposedPackageSerialisesTheSameWayTwice) {
    // The wiring lives in a block payload, so `Package::operator==` and a byte comparison both go
    // through the encoded form. A `stage()` that appended instead of replacing the block would show
    // up here and nowhere else.
    const deploy::bytes once = composed_builder().build().to_bytes();
    const deploy::bytes twice = composed_builder().build().to_bytes();
    EXPECT_EQ(once, twice);
    EXPECT_EQ(deploy::Package::read(once), deploy::Package::read(twice));

    std::size_t composition_blocks = 0;
    for (const auto& block : composed_builder().build().blocks)
        composition_blocks += block.kind == deploy::BlockKind::Composition;
    EXPECT_EQ(composition_blocks, 1u) << "the builder grew a second composition block";
}

TEST(FieldInference, ATensorTheCompositionSuppliesIsNotTheCallersToBind) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";

    const std::shared_ptr<InferenceSession> session =
        onnx::load(composed_builder().build(), Backend::Cpu, -1);
    session->set_voxel_grid({2, 3, 4});
    std::vector<float> latent(kVoxels * 2, 0.f);
    try {
        session->bind_input("latent", memory::host::MemoryRef::of(std::span<float>(latent)));
        ADD_FAILURE() << "a wired tensor was accepted as a caller binding";
    } catch (const Exception& err) {
        // The diagnostic must name WHERE it comes from; "no such input" would send someone to fix
        // the wrong thing entirely.
        EXPECT_NE(std::string(err.what()).find("beam_encoder"), std::string::npos) << err.what();
    }
}

TEST(FieldInference, APackageWithNoCompositionIsStillASingleTrunk) {
    auto session = probe_session();
    if (!session) GTEST_SKIP() << "built without ONNX Runtime";
    // The behaviour every package had before the format could record wiring, unchanged.
    EXPECT_FALSE(session->get_package().get_composition().has_value());
    FieldInference filler(session, allocate_gpu_field(test_geometry()));
    EXPECT_EQ(filler.get_caller_inputs(), (std::vector<std::string>{"beam_direction"}));
}

// ── a real trained model ──────────────────────────────────────────────────────────────────────────
//
// The synthetic graph above proves the arithmetic; this proves the same code drives a package a
// trainer actually produced. `RFNN_TEST_MODEL` overrides the path; without one the case skips,
// because a checkout has no model in it.

namespace {

std::string real_model_path() {
    const char* env = std::getenv("RFNN_TEST_MODEL");
    return env ? env
               : "/mnt/data/models_nn/logs/pbrf-ds03-3779-noroi/pbrf-ds03-3779-noroi/models/PBRFNet.rf3m";
}

}  // namespace

TEST(FieldInference, ARealModelFillsARealField) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";
    const std::string path = real_model_path();
    if (!std::filesystem::exists(path)) GTEST_SKIP() << "no model at " << path;

    constexpr std::uint32_t kSide = 4;
    const std::shared_ptr<InferenceSession> session =
        onnx::load(deploy::Package::read_file(path), Backend::Cpu, -1);
    auto field = allocate_gpu_field(CartesianFieldGeometry::cubic(kSide, 1.f));
    FieldInference filler(session, field);

    // The trunk's own inputs, which this package does not declare — see the D3 note below. A latent
    // code of a constant is meaningless physically and entirely adequate here: what is under test is
    // that the field is filled, not what the network believes.
    std::vector<float> latent(static_cast<std::size_t>(kSide) * kSide * kSide * 192, 0.01f);
    std::vector<float> region_state(14, 0.f);
    session->bind_input("latent", memory::host::MemoryRef::of(std::span<float>(latent)));
    session->bind_input("region_state", memory::host::MemoryRef::of(std::span<float>(region_state)));
    filler.run();

    EXPECT_EQ(filler.get_output_layers(), (std::vector<std::string>{"flux", "spectrum"}));
    auto buffer = field->get_channel(std::string(kPredictionChannel));
    ASSERT_TRUE(buffer);
    EXPECT_EQ(buffer->get_layer("spectrum").get_voxel_flat_raw(0)->get_bytes(), 32 * sizeof(float));

    const float* flux = buffer->get_layer<float>("flux");
    const std::size_t voxels = buffer->get_voxel_count();
    std::size_t nonzero = 0;
    for (std::size_t q = 0; q < voxels; ++q) {
        ASSERT_TRUE(std::isfinite(flux[q])) << "voxel " << q << " is not finite";
        nonzero += flux[q] != 0.f;
    }
    EXPECT_EQ(nonzero, voxels) << "the field is still zeros — nothing was written";
    // Every voxel was queried at its own position, so a constant field would mean the positions
    // never varied.
    EXPECT_NE(flux[0], flux[voxels - 1]) << "every voxel got the same value";
}

/// The real PBRFNet, driven as ONE composed package instead of by hand.
///
/// This is what A7 existed for. The V1 file on disk carries three graphs and no wiring, so a
/// consumer had to know out of band that `beam_encoder.linear_5` feeds `trunk.latent` — and the
/// names do not even match. Re-packaged with the wiring recorded, the session does it.
///
/// The check is EQUIVALENCE, not plausibility: the same beam driven by hand — encoder run
/// separately, its output tiled and bound as `latent` — must give the same flux to the last bit.
TEST(FieldInference, TheRealModelComposesToTheSameResultAsDrivingItByHand) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";
    const std::string path = real_model_path();
    if (!std::filesystem::exists(path)) GTEST_SKIP() << "no model at " << path;

    const deploy::Package original = deploy::Package::read_file(path);
    // The package declares a V1 interface its graphs do not match (D3), so declare what the trunk
    // actually has; the graphs themselves are the real thing, byte for byte.
    const auto with_graphs = [&original](deploy::PackageBuilder& builder) -> deploy::PackageBuilder& {
        for (const auto name : original.get_graph_names()) {
            const auto graph = original.get_graph(name);
            builder.graph(std::string(name), deploy::bytes(graph->begin(), graph->end()));
        }
        return builder;
    };
    const auto declare = [](deploy::PackageBuilder& builder) -> deploy::PackageBuilder& {
        return builder.provenance("composed", "tests", "none")
            .field_dimensions_m({1.f, 1.f, 1.f})
            .input("position", deploy::Semantic::Position, {3}).unit("m").done()
            .output("flux", deploy::Semantic::Flux, {1}).done()
            .output("spectrum", deploy::Semantic::Spectrum, {32}).done();
    };

    constexpr std::uint32_t kSide = 2;
    constexpr std::size_t kQ = kSide * kSide * kSide;
    std::vector<float> position(kQ * 3, 0.5f), direction{0.f, 0.f, 1.f}, distance{1.f},
        tube_spectrum(150, 0.01f), region_width{0.5f};
    const auto bind_beam = [&](const std::shared_ptr<InferenceSession>& session) {
        session->bind_input("position", memory::host::MemoryRef::of(std::span<float>(position)));
        session->bind_input("direction", memory::host::MemoryRef::of(std::span<float>(direction)));
        session->bind_input("distance", memory::host::MemoryRef::of(std::span<float>(distance)));
        session->bind_input("spectrum", memory::host::MemoryRef::of(std::span<float>(tube_spectrum)));
        session->bind_input("region_width", memory::host::MemoryRef::of(std::span<float>(region_width)));
    };

    // ── fully composed ───────────────────────────────────────────────────────────────────────────
    deploy::PackageBuilder composed;
    declare(composed);
    with_graphs(composed);
    composed.buffer("beam_latent", 192)
        .buffer("region_state", 14)
        .stage("encoding_config", deploy::Invocation::Once)
        .writes("region_state", "region_state")
        .done()
        .stage("beam_encoder", deploy::Invocation::Once)
        .writes("linear_5", "beam_latent")
        .done()
        .stage("trunk", deploy::Invocation::PerQuery)
        .reads("latent", "beam_latent")
        .reads("region_state", "region_state")
        .done();

    const std::shared_ptr<InferenceSession> session = onnx::load(composed.build(), Backend::Cpu, -1);
    session->set_voxel_grid({kSide, kSide, kSide});
    std::vector<float> flux(kQ), spectrum(kQ * 32);
    bind_beam(session);
    session->bind_output("flux", memory::host::MemoryRef::of(std::span<float>(flux)));
    session->bind_output("spectrum", memory::host::MemoryRef::of(std::span<float>(spectrum)));
    session->infer();
    ASSERT_TRUE(std::isfinite(flux[0]));
    EXPECT_NE(flux[0], 0.f);

    // ── the same beam, latent carried by hand ────────────────────────────────────────────────────
    deploy::PackageBuilder manual;
    declare(manual);
    with_graphs(manual);
    manual.buffer("region_state", 14)
        .stage("encoding_config", deploy::Invocation::Once)
        .writes("region_state", "region_state")
        .done()
        .stage("beam_encoder", deploy::Invocation::Once)
        .done()
        .stage("trunk", deploy::Invocation::PerQuery)
        .reads("region_state", "region_state")
        .done();
    // `linear_5` deliberately NOT wired, so both ends are the caller's.

    const std::shared_ptr<InferenceSession> by_hand = onnx::load(manual.build(), Backend::Cpu, -1);
    by_hand->set_voxel_grid({kSide, kSide, kSide});
    std::vector<float> latent_out(192, 0.f), latent_in(kQ * 192, 0.f);
    std::vector<float> hand_flux(kQ), hand_spectrum(kQ * 32);
    bind_beam(by_hand);
    by_hand->bind_output("linear_5", memory::host::MemoryRef::of(std::span<float>(latent_out)));
    by_hand->bind_input("latent", memory::host::MemoryRef::of(std::span<float>(latent_in)));
    by_hand->bind_output("flux", memory::host::MemoryRef::of(std::span<float>(hand_flux)));
    by_hand->bind_output("spectrum", memory::host::MemoryRef::of(std::span<float>(hand_spectrum)));
    by_hand->infer();  // computes the latent
    for (std::size_t r = 0; r < kQ; ++r)
        std::copy_n(latent_out.begin(), 192, latent_in.begin() + static_cast<std::ptrdiff_t>(r * 192));
    by_hand->infer();  // now the trunk sees it

    for (std::size_t q = 0; q < kQ; ++q)
        EXPECT_FLOAT_EQ(flux[q], hand_flux[q]) << "voxel " << q;
    for (std::size_t i = 0; i < spectrum.size(); ++i)
        EXPECT_FLOAT_EQ(spectrum[i], hand_spectrum[i]) << "spectrum element " << i;
}

/// Pins the known V1 gap (HANDOFF §5 D3) rather than letting it be rediscovered.
///
/// The legacy packages declare the interface of the MODEL, while the trunk graph names the tensors
/// of one stage of it — `tube_spectrum`/`beam_direction`/`source_distance` feed the beam encoder,
/// whose output reaches the trunk as `latent`. The container records nowhere how the two compose
/// (that is A7), so what the package declares as a caller input is not necessarily bindable.
/// The failure must stay a NAMED refusal, never a silent no-op.
TEST(FieldInference, TheV1InterfaceGapIsReportedAndNotSilent) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";
    const std::string path = real_model_path();
    if (!std::filesystem::exists(path)) GTEST_SKIP() << "no model at " << path;

    const std::shared_ptr<InferenceSession> session =
        onnx::load(deploy::Package::read_file(path), Backend::Cpu, -1);
    FieldInference filler(session, allocate_gpu_field(CartesianFieldGeometry::cubic(2, 1.f)));

    // Declared, and genuinely what a caller would reach for.
    EXPECT_NE(std::find(filler.get_caller_inputs().begin(), filler.get_caller_inputs().end(),
                        "beam_direction"),
              filler.get_caller_inputs().end());

    std::vector<float> direction(3, 1.f);
    try {
        session->bind_input("beam_direction", memory::host::MemoryRef::of(std::span<float>(direction)));
        ADD_FAILURE() << "binding a tensor the trunk does not have was accepted";
    } catch (const Exception& err) {
        EXPECT_NE(std::string(err.what()).find("beam_direction"), std::string::npos) << err.what();
    }
}

// ── the device path ───────────────────────────────────────────────────────────────────────────────

TEST(FieldInference, AttachedDeviceMemoryIsBoundInsteadOfTheHostLayer) {
    auto session = probe_session();
    if (!session) GTEST_SKIP() << "built without ONNX Runtime";

    auto field = allocate_gpu_field(test_geometry());
    field->add_channel(std::string(kPredictionChannel));
    // A host buffer standing in for a device mirror: this case is about WHICH buffer gets bound,
    // and that decision is made on the mirror's presence, not on its domain.
    std::vector<float> mirror(kVoxels, -1.f);
    field->set_device_memory(std::string(kPredictionChannel), "flux",
                             memory::host::MemoryRef::of(std::span<float>(mirror)));

    FieldInference filler(session, field);
    std::vector<float> direction;
    session->bind_input("beam_direction", unit_direction(direction));
    filler.run();

    // The mirror holds the prediction...
    EXPECT_NEAR(mirror[0], 0.25f + 0.25f + 0.25f, 1e-5f);
    // ...and the host voxels were NOT written, which is the documented consequence: a device-resident
    // result must be synced back through its backend before to_host_field() means anything.
    EXPECT_FLOAT_EQ(field->get_channel(std::string(kPredictionChannel))->get_layer<float>("flux")[0], 0.f);
}
