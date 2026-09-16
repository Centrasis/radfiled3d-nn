// Round-trip tests for the RF3M container.
//
// These are the tests that hold defect D1 (requirements.md §2) closed: the original writes the
// byte layout once and reads it back in three separate hand-written walks, so a field added to one
// and forgotten in another desynchronises the parse silently. Here every walk goes through the same
// encode/decode pair, and a round trip proves the pair agrees with itself.
//
// This binary links RadFiled3D::nn::deploy only: no RadFiled3D, no GPU.
#include <RadFiled3D/nn/deploy.hpp>
#include <RadFiled3D/nn/deploy/version.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

using namespace RadFiled3D::nn::deploy;

namespace {

bytes b(const char* s) { return bytes(s, s + std::char_traits<char>::length(s)); }

/// A package shaped like what the training framework exports for a TPBRFNet: a per-voxel implicit
/// field taking the beam parameters plus a patient translation, emitting flux and a spectrum.
PackageBuilder sample_builder() {
    constexpr double pi = std::numbers::pi;
    PackageBuilder builder;
    builder.provenance("xray-scatter-v3", "radfield3d-nn 2.1", "G4EmStandardPhysics_option4")
        .created("2026-08-31T08:49:10Z")
        .field_dimensions_m({1.f, 1.f, 1.f})
        .input("position", Semantic::Position, {3}).unit("m").normalizer(Linear01{0.0, 1.0}).done()
        .input("beam_direction", Semantic::BeamDirection, {2}).unit("rad")
            .range(MinMax{-pi, pi}).normalizer(LinearSym{-pi, pi}).done()
        .input("source_distance", Semantic::SourceDistance, {1}).unit("m").range(MinMax{0.5, 3.0}).done()
        .input("patient_translation", Semantic::PatientTranslation, {3}).unit("m").range(MinMax{-0.5, 0.5}).done()
        .input("tube_spectrum", Semantic::TubeSpectrum, {128}).unit("eV")
            .range(Histogram{0.0, 150'000.0, 1171.875}).done()
        .output("flux", Semantic::Flux, {1}).normalizer(LogScale{1e-12, 30.0}).done()
        .output("spectrum", Semantic::Spectrum, {128}).unit("eV").done()
        .graph("trunk", b("trunk-onnx-bytes"))
        .graph("beam_encoder", b("encoder-onnx-bytes"))
        .metric("air_kerma_smape", 0.0431)
        .metric("gamma_pass_rate", 0.981);
    return builder;
}

Package sample_package() { return sample_builder().build(); }

std::uint32_t le32(const bytes& v, std::size_t at) {
    return v[at] | (v[at + 1] << 8) | (v[at + 2] << 16) | (std::uint32_t(v[at + 3]) << 24);
}
std::uint64_t le64(const bytes& v, std::size_t at) {
    std::uint64_t out = 0;
    for (int i = 7; i >= 0; --i) out = (out << 8) | v[at + static_cast<std::size_t>(i)];
    return out;
}

}  // namespace

TEST(Rf3mRoundTrip, RoundTripsThroughBytes) {
    const Package original = sample_package();
    const Package decoded = Package::read(original.to_bytes());
    EXPECT_EQ(original, decoded);
}

TEST(Rf3mRoundTrip, TheHeaderStartsWithMagicAndAFileVersion) {
    const bytes data = sample_package().to_bytes();
    EXPECT_EQ(std::string(data.begin(), data.begin() + 4), "RF3M");
    EXPECT_EQ(le32(data, 4), 1u);
}

TEST(Rf3mRoundTrip, MetadataReadSkipsThePayloads) {
    const Package original = sample_package();
    const Metadata md = Package::read_metadata(original.to_bytes());
    EXPECT_EQ(md.provenance, original.provenance);
    EXPECT_EQ(md.io, original.io);
    EXPECT_EQ(md.metrics, original.metrics);
    // Every block is described — kind, name, size — without a payload byte being read.
    ASSERT_EQ(md.blocks.size(), 2u);
    EXPECT_EQ(md.blocks[0].name, "trunk");
    EXPECT_EQ(md.blocks[1].name, "beam_encoder");
    for (const auto& blk : md.blocks) EXPECT_EQ(blk.kind, BlockKind(BlockKind::Onnx));
    EXPECT_EQ(md.blocks[0].payload_bytes, std::string("trunk-onnx-bytes").size());
}

TEST(Rf3mRoundTrip, ACorruptPackageIsRefused) {
    bytes data = sample_package().to_bytes();
    data.back() ^= 0xff;
    try {
        (void)Package::read(data);
        FAIL() << "a corrupt package was accepted";
    } catch (const RadFiled3D::nn::Exception& err) {
        EXPECT_EQ(err.get_kind(), RadFiled3D::nn::ErrorKind::DigestMismatch);
    }
}

TEST(Rf3mRoundTrip, AFutureVersionIsReportedByNumber) {
    bytes data = sample_package().to_bytes();
    data[4] = 7;
    try {
        (void)Package::read_metadata(data);
        FAIL();
    } catch (const RadFiled3D::nn::Exception& err) {
        EXPECT_EQ(err.get_kind(), RadFiled3D::nn::ErrorKind::UnsupportedVersion);
        EXPECT_NE(std::string(err.what()).find("7"), std::string::npos);
    }
}

/// The point of the redesign: a model with an output this build has never heard of round-trips
/// and stays fully describable. In the original the equivalent would be a reserved interface bit,
/// which the reader rejects outright.
TEST(Rf3mRoundTrip, AnUnknownOutputSemanticSurvives) {
    PackageBuilder builder;
    builder.field_dimensions_m({1.f, 1.f, 1.f})
        .input("position", Semantic::Position, {3}).done()
        .output("flux", Semantic::Flux, {1}).done()
        .output("direction_distribution", Semantic::from_name("direction_distribution"), {16, 32}).unit("sr^-1").done()
        .graph("trunk", b("onnx"));
    const Package package = builder.build();

    const Package decoded = Package::read(package.to_bytes());
    EXPECT_EQ(decoded, package);

    const TensorDescriptor* extra = decoded.get_tensor("direction_distribution");
    ASSERT_NE(extra, nullptr);
    EXPECT_EQ(extra->shape, (std::vector<std::uint32_t>{16, 32}));
    EXPECT_FALSE(extra->semantic.is_known());
    EXPECT_EQ(decoded.get_unknown_semantics(), (std::vector<std::string_view>{"direction_distribution"}));
    // The resolution travels with the tensor, so nothing else in the format had to change.
    EXPECT_EQ(extra->get_elements(), 16u * 32u);
}

/// A block kind written by a newer producer must survive a read/write cycle untouched, or an old
/// tool would silently strip an accelerator's code on rewrite.
TEST(Rf3mRoundTrip, AnUnknownBlockKindIsPreserved) {
    Package package = sample_package();
    package.blocks.push_back(Block{BlockKind::from_name("metal_msl"), "trunk", "apple8", {1, 2, 3, 4, 5}});

    const Package decoded = Package::read(package.to_bytes());
    EXPECT_EQ(decoded, package);
    const Block& extra = decoded.blocks.back();
    EXPECT_FALSE(extra.kind.is_known());
    EXPECT_EQ(extra.kind.get_name(), "metal_msl");
    // It is carried, described in a scan, and simply never selected.
    EXPECT_EQ(decoded.select_block("trunk", "cuda", "sm_90"), nullptr);
}

TEST(Rf3mRoundTrip, BlocksSelectByKindAndArchitecture) {
    PackageBuilder builder = sample_builder();
    builder.block(BlockKind::CudaCubin, "trunk", "sm_90", {0xde, 0xad})
        .block(BlockKind::CudaPtx, "trunk", "", {0xbe, 0xef});  // driver JITs it: any arch
    const Package package = builder.build();

    // An exact architecture match wins over the agnostic PTX ...
    ASSERT_NE(package.select_block("trunk", "cuda", "sm_90"), nullptr);
    EXPECT_EQ(package.select_block("trunk", "cuda", "sm_90")->kind, BlockKind(BlockKind::CudaCubin));
    // ... and on another NVIDIA architecture the PTX still serves.
    ASSERT_NE(package.select_block("trunk", "cuda", "sm_80"), nullptr);
    EXPECT_EQ(package.select_block("trunk", "cuda", "sm_80")->kind, BlockKind(BlockKind::CudaPtx));
    // On an AMD card neither applies, and the ONNX graph is the fallback that always exists.
    EXPECT_EQ(package.select_block("trunk", "rocm", "gfx1100"), nullptr);
    EXPECT_TRUE(package.get_graph("trunk").has_value());

    EXPECT_EQ(Package::read(package.to_bytes()), package);
}

TEST(Rf3mRoundTrip, APackageWithNoRunnableTrunkIsRefused) {
    // SOMETHING must run the model. A cubin qualifies — a package may deliberately ship a trunk only
    // as vendor code, and the consequence lands at load as a sentence naming the stage and the
    // device (`require_runnable_on`) rather than as a refusal to write the file at all.
    PackageBuilder specialised;
    specialised.input("position", Semantic::Position, {3}).done()
        .output("flux", Semantic::Flux, {1}).done()
        .block(BlockKind::CudaCubin, "trunk", "sm_90", {0});
    const Package package = specialised.build();
    EXPECT_FALSE(package.is_portable());
    EXPECT_NO_THROW(package.require_runnable_on("cuda", "sm_90"));
    EXPECT_THROW(package.require_runnable_on("cuda", "sm_86"), RadFiled3D::nn::Exception);

    // A package with NO executable trunk at all is still refused: the composition block describes
    // how to run a model and is not itself a way to run one.
    PackageBuilder empty;
    empty.input("position", Semantic::Position, {3}).done().output("flux", Semantic::Flux, {1}).done();
    try {
        (void)empty.build();
        FAIL() << "a package must carry the model in some runnable form";
    } catch (const RadFiled3D::nn::Exception& err) {
        EXPECT_NE(std::string(err.what()).find("trunk"), std::string::npos);
    }
}

/// A fixed voxelization on a point-field model would freeze a resolution the caller is supposed to
/// choose at inference time (R-F4).
TEST(Rf3mRoundTrip, AVoxelizationIsRefusedOnAPerVoxelModel) {
    PackageBuilder point;
    point.field_dimensions_m({1.f, 1.f, 1.f})
        .voxelization({64, 64, 64}, {1.f / 64, 1.f / 64, 1.f / 64})
        .input("position", Semantic::Position, {3}).done()
        .output("flux", Semantic::Flux, {1}).done()
        .graph("trunk", b("onnx"));
    try {
        (void)point.build();
        FAIL() << "a point field may not fix a grid";
    } catch (const RadFiled3D::nn::Exception& err) {
        EXPECT_NE(std::string(err.what()).find("voxelization"), std::string::npos);
    }

    // The same voxelization on a whole-volume model is fine: there the grid IS the architecture.
    PackageBuilder volume;
    volume.field_dimensions_m({1.f, 1.f, 1.f})
        .voxelization({64, 64, 64}, {1.f / 64, 1.f / 64, 1.f / 64})
        .input("beam_direction", Semantic::BeamDirection, {2}).done()
        .output("flux", Semantic::Flux, {64, 64, 64}).done()
        .graph("trunk", b("onnx"));
    const Package package = volume.build();
    EXPECT_FALSE(package.is_voxelwise());
    ASSERT_TRUE(package.geometry.voxelization.has_value());
    EXPECT_EQ(package.geometry.voxelization->voxel_counts, (std::array<std::uint32_t, 3>{64, 64, 64}));
}

TEST(Rf3mRoundTrip, DeclaringBothSourceParameterisationsIsRefused) {
    PackageBuilder builder;
    builder.input("source_distance", Semantic::SourceDistance, {1}).done()
        .input("source_origin", Semantic::SourceOrigin, {3}).done()
        .output("flux", Semantic::Flux, {1}).done()
        .graph("trunk", b("onnx"));
    try {
        (void)builder.build();
        FAIL();
    } catch (const RadFiled3D::nn::Exception& err) {
        EXPECT_NE(std::string(err.what()).find("alternatives"), std::string::npos);
    }
}

TEST(Rf3mRoundTrip, RolesPartitionTheDescriptors) {
    const Package package = sample_package();
    EXPECT_EQ(package.get_inputs().size(), 5u);
    EXPECT_EQ(package.get_outputs().size(), 2u);
    EXPECT_TRUE(package.is_voxelwise());
    const TensorDescriptor* spectrum = package.get_tensor_with(Role::Output, Semantic::Spectrum);
    ASSERT_NE(spectrum, nullptr);
    EXPECT_EQ(spectrum->get_elements(), 128u);
}

TEST(Rf3mRoundTrip, NormalizersConvertMetricValues) {
    constexpr double pi = std::numbers::pi;
    const Package package = sample_package();
    const TensorDescriptor* direction = package.get_tensor("beam_direction");
    ASSERT_NE(direction, nullptr);
    // The caller passes radians; the package says how they reach the graph.
    EXPECT_EQ(apply(direction->normalizer, -pi), std::optional<double>(-1.0));
    EXPECT_EQ(apply(direction->normalizer, pi), std::optional<double>(1.0));
    // A range knows what it was trained on, so an out-of-distribution query can be reported.
    EXPECT_EQ(contains(direction->range, 0.0), std::optional<bool>(true));
    EXPECT_EQ(contains(direction->range, 4.0), std::optional<bool>(false));
    // A normalizer this build does not implement answers nothing rather than a plausible number.
    EXPECT_EQ(apply(CustomNormalizer{"quantile", {0.1}}, 1.0), std::nullopt);
}

/// `invert` is `apply` run backwards, and the property that says so is the round trip.
///
/// Checked over the whole implemented set and a spread of values rather than on one hand-computed
/// case per kind: an inverse with a sign or a factor wrong still reproduces a single well-chosen
/// point, and would pass that.
TEST(Rf3mRoundTrip, InvertUndoesApplyForEveryImplementedNormalizer) {
    const std::vector<Normalizer> normalizers = {
        Identity{},
        Linear01{0.0, 1.0},
        Linear01{-0.5, 2.5},
        LinearSym{-std::numbers::pi, std::numbers::pi},
        LinearSym{0.0, 150'000.0},
        LogScale{1e-12, 30.0},
        Asinh{1e-6},
        Asinh{2.0},
    };
    // Positive throughout: LogScale takes a logarithm, so its domain is `v > -epsilon`, and a
    // negative flux is not a value any of these normalizers is asked to carry.
    const std::vector<double> values = {0.0, 1e-9, 0.25, 1.0, 7.5, 1234.5};

    for (const Normalizer& n : normalizers) {
        ASSERT_TRUE(is_invertible(n)) << get_name(n);
        for (const double metric : values) {
            const std::optional<double> graph = apply(n, metric);
            ASSERT_TRUE(graph.has_value()) << get_name(n) << " could not encode " << metric;
            const std::optional<double> back = invert(n, *graph);
            ASSERT_TRUE(back.has_value()) << get_name(n) << " could not decode " << *graph;
            // Relative, because these span twelve decades and an absolute epsilon would be either
            // vacuous at the top or impossible at the bottom.
            EXPECT_NEAR(*back, metric, 1e-9 * std::max(1.0, std::abs(metric)))
                << get_name(n) << " round trip of " << metric << " via " << *graph;
        }
    }
}

TEST(Rf3mRoundTrip, AnInverseThatCannotProduceANumberAnswersNothing) {
    // Unknown arithmetic: the name survives a round trip and can be reported, but nothing here can
    // compute with it — in either direction.
    const Normalizer custom = CustomNormalizer{"quantile", {0.1}};
    EXPECT_FALSE(is_invertible(custom));
    EXPECT_EQ(invert(custom, 0.5), std::nullopt);

    // A degenerate parameterisation has no inverse either. Forward it is a division by zero;
    // backward a multiplication by zero, which would answer a plausible `min` for every input.
    EXPECT_EQ(invert(Linear01{1.0, 1.0}, 0.5), std::nullopt);
    EXPECT_EQ(invert(LinearSym{2.0, 2.0}, 0.5), std::nullopt);
    EXPECT_EQ(invert(LogScale{1e-12, 0.0}, 0.5), std::nullopt);
    EXPECT_EQ(invert(Asinh{0.0}, 0.5), std::nullopt);

    // Going back through `exp` overflows for a large enough graph value. An infinite flux is not a
    // reading, so it is refused rather than written into a field.
    EXPECT_EQ(invert(LogScale{0.0, 30.0}, 1000.0), std::nullopt);
    // And forward, a logarithm of a negative value is NaN — the same silently-wrong number.
    EXPECT_EQ(apply(LogScale{0.0, 30.0}, -1.0), std::nullopt);
}

// ── graph composition ─────────────────────────────────────────────────────────────────────────────
//
// A real model is several graphs. `PBRFNet` is `beam_encoder` -> `trunk` with an `encoding_config`
// beside it, and until this existed the container recorded nowhere how they connect — the packages
// on disk name the encoder's output `linear_5` while the trunk's input is `latent`, so a consumer
// could not even guess the wiring from the names.

namespace {

/// A package wired the way a real PBRFNet is.
PackageBuilder composed_builder() {
    PackageBuilder builder = sample_builder();
    builder.graph("encoding_config", b("config-onnx"))
        // The memory between the stages. `elements` is PER ROW; how many rows each holds is worked
        // out when the graphs are wired, from the shapes they declare.
        .buffer("beam_latent", 192)
        .buffer("region_state", 14)
        // Execution order, first to last.
        .stage("encoding_config", Invocation::Once)
        .writes("region_state", "region_state")
        .done()
        .stage("beam_encoder", Invocation::Once)
        // The port the names alone would NOT give you: the exporter that wrote the packages on
        // disk called this output `linear_5`, and the trunk calls its input `latent`. The buffer is
        // what reconciles them.
        .writes("linear_5", "beam_latent")
        .done()
        .stage("trunk", Invocation::PerQuery)
        .reads("latent", "beam_latent")
        .reads("region_state", "region_state")
        .done();
    return builder;
}

}  // namespace

TEST(Rf3mRoundTrip, ACompositionSurvivesARoundTrip) {
    const Package original = composed_builder().build();
    const Package back = Package::read(original.to_bytes());

    const auto composition = back.get_composition();
    ASSERT_TRUE(composition.has_value());
    EXPECT_EQ(*composition, *original.get_composition());
    EXPECT_EQ(composition->stages.size(), 3u);
    EXPECT_EQ(composition->stages.back().name, "trunk");
    EXPECT_EQ(composition->stages.back().invocation, Invocation::PerQuery);
    EXPECT_EQ(composition->get_stage("beam_encoder")->invocation, Invocation::Once);

    // The wiring a consumer needs, by name: which stage fills the buffer the trunk reads.
    const Stage* trunk = composition->get_stage("trunk");
    ASSERT_NE(trunk, nullptr);
    ASSERT_EQ(trunk->reads.size(), 2u);
    EXPECT_EQ(trunk->reads[0].tensor, "latent");
    EXPECT_EQ(trunk->reads[0].buffer, "beam_latent");
    const Stage* writer = composition->get_writer_of("beam_latent");
    ASSERT_NE(writer, nullptr);
    EXPECT_EQ(writer->name, "beam_encoder");
    EXPECT_EQ(writer->writes[0].tensor, "linear_5");
    // An input nothing feeds is the caller's to bind.
    EXPECT_EQ(composition->get_buffer("nothing_declares_this"), nullptr);
    EXPECT_EQ(composition->get_buffer("beam_latent")->elements, 192u);
}

TEST(Rf3mRoundTrip, APackageWithoutACompositionIsASingleTrunk) {
    // Every package written before this existed, and every single-graph model still. Absent is a
    // meaning, not a missing field.
    EXPECT_FALSE(sample_package().get_composition().has_value());
}

/// The reason the wiring is a BLOCK and not a metadata field.
///
/// Rule 5 says a block a build does not recognise is skipped by its length and written back
/// verbatim. The metadata region carries no such guarantee — a reader takes it by length and stops
/// at the last field it knows — so a composition stored there would be silently dropped by any tool
/// that had not heard of it, leaving a package whose graphs can no longer be connected.
TEST(Rf3mRoundTrip, ACompositionIsPreservedByATOolThatDoesNotUnderstandIt) {
    const bytes original = composed_builder().build().to_bytes();

    // Stand in for an older build: read the package, touch something unrelated, write it back.
    Package rewritten = Package::read(original);
    rewritten.metrics["rewritten_by"] = 1.0;
    const Package back = Package::read(rewritten.to_bytes());

    ASSERT_TRUE(back.get_composition().has_value());
    EXPECT_EQ(*back.get_composition(), *Package::read(original).get_composition());
}

TEST(Rf3mRoundTrip, ACompositionBlockIsNotAWayToRunTheModel) {
    // It describes how the executable blocks fit together; it is not one of them. A package whose
    // only non-ONNX block is a composition still needs its trunk.
    const Package package = composed_builder().build();
    for (const auto& block : package.blocks)
        if (block.kind == BlockKind::Composition) {
            EXPECT_FALSE(block.kind.is_executable());
            EXPECT_TRUE(block.kind.is_known());
        }
    // And it is not mistaken for a graph, nor for something a stage could name.
    const auto graphs = package.get_graph_names();
    EXPECT_EQ(std::find(graphs.begin(), graphs.end(), "composition"), graphs.end());
    const auto executable = package.get_executable_block_names();
    EXPECT_EQ(std::find(executable.begin(), executable.end(), "composition"), executable.end());
}

// ── specialised code per stage ────────────────────────────────────────────────────────────────────
//
// A stage is implemented by whatever BLOCKS carry its name. The composition does not list the
// variants — that would be a second source of truth about what the package holds — so shipping a
// CUDA kernel for one stage is `block()` plus a launch descriptor, and nothing else changes.

namespace {

/// A package whose `hashgrid` stage exists ONLY as vendor-specific code: PTX for NVIDIA, an HSACO
/// object for AMD, SPIR-V for Intel. No ONNX form, which is exactly the case that must fail by name
/// on hardware none of them runs on.
PackageBuilder specialised_builder() {
    PackageBuilder builder = sample_builder();
    builder.block(BlockKind::CudaPtx, "hashgrid", "", b("ptx-source"))
        .block(BlockKind::Hsaco, "hashgrid", "gfx1100", b("amd-code-object"))
        .block(BlockKind::SpirV, "hashgrid", "", b("spirv-words"))
        .buffer("encoded", 32)
        .stage("hashgrid", Invocation::Once)
        .writes("features", "encoded")
        .launch(BlockKind::CudaPtx, "", KernelLaunch{"hashgrid_encode", {128, 1, 1}, {0, 0, 0}, 0})
        .launch(BlockKind::Hsaco, "gfx1100", KernelLaunch{"hashgrid_encode_amd", {64, 1, 1}, {0, 0, 0}, 0})
        .done()
        .stage("trunk", Invocation::PerQuery)
        .reads("features", "encoded")
        .done();
    return builder;
}

}  // namespace

TEST(Rf3mRoundTrip, AStageMayBeImplementedOnlyBySpecialisedCode) {
    const Package package = Package::read(specialised_builder().build().to_bytes());

    // It validates: nothing requires a stage to have a portable form.
    const auto composition = package.get_composition();
    ASSERT_TRUE(composition.has_value());
    EXPECT_EQ(composition->get_stage("hashgrid")->launches.size(), 2u);
    // But the package knows it is not portable, which is the question a producer wants answered.
    EXPECT_FALSE(package.is_portable());
    EXPECT_TRUE(sample_package().is_portable());

    // The variants live in the BLOCKS, found by the stage's name, and selection is by device.
    EXPECT_EQ(package.select_block("hashgrid", "cuda", "sm_86")->kind, BlockKind(BlockKind::CudaPtx));
    EXPECT_EQ(package.select_block("hashgrid", "rocm", "gfx1100")->kind, BlockKind(BlockKind::Hsaco));
    // An AMD card the object was not built for: the HSACO names an architecture and it is not this
    // one, so nothing runs it.
    EXPECT_EQ(package.select_block("hashgrid", "rocm", "gfx90a"), nullptr);
}

TEST(Rf3mRoundTrip, UnsupportedHardwareIsRefusedByStageName) {
    const Package package = specialised_builder().build();

    // NVIDIA and the right AMD card both work.
    EXPECT_NO_THROW(package.require_runnable_on("cuda", "sm_86"));
    EXPECT_NO_THROW(package.require_runnable_on("rocm", "gfx1100"));

    // The CPU provider has no vendor code at all, and the message has to say WHICH stage and WHAT
    // the package carries — a user who only learns "unsupported" learns nothing actionable.
    try {
        package.require_runnable_on("cpu", "");
        FAIL() << "a package with a vendor-only stage must not load on the CPU provider";
    } catch (const RadFiled3D::nn::Exception& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("hashgrid"), std::string::npos) << what;
        EXPECT_NE(what.find("cuda_ptx"), std::string::npos) << what;
        EXPECT_NE(what.find("hsaco(gfx1100)"), std::string::npos) << what;
        EXPECT_NE(what.find("spirv"), std::string::npos) << what;
        EXPECT_NE(what.find("cpu"), std::string::npos) << what;
        // The trunk is portable, so it must NOT be blamed.
        EXPECT_EQ(what.find("`trunk`"), std::string::npos) << what;
    }
}

TEST(Rf3mRoundTrip, ALaunchDescriptorIsChosenTheWayABlockIs) {
    const Package package = Package::read(specialised_builder().build().to_bytes());
    // Held: `get_composition` returns by value, so a pointer into the temporary would dangle.
    const auto composition = package.get_composition();
    const Stage* hashgrid = composition->get_stage("hashgrid");

    // Architecture-agnostic PTX serves any NVIDIA card.
    ASSERT_NE(hashgrid->get_launch_for("cuda_ptx", "sm_86"), nullptr);
    EXPECT_EQ(hashgrid->get_launch_for("cuda_ptx", "sm_86")->entry_point, "hashgrid_encode");
    EXPECT_EQ(hashgrid->get_launch_for("cuda_ptx", "sm_86")->block_size[0], 128u);
    // The AMD entry point differs, which is why launches are per implementation and not per stage.
    EXPECT_EQ(hashgrid->get_launch_for("hsaco", "gfx1100")->entry_point, "hashgrid_encode_amd");
    // SPIR-V ships without one; there is nothing to launch it with here.
    EXPECT_EQ(hashgrid->get_launch_for("spirv", ""), nullptr);

    // A zero grid component is derived from the work, so one descriptor serves every resolution.
    EXPECT_EQ(hashgrid->get_launch_for("cuda_ptx", "")->get_grid_for(1000)[0], 8u);  // ceil(1000/128)
    EXPECT_EQ(hashgrid->get_launch_for("cuda_ptx", "")->get_grid_for(1000)[1], 1u);
}

/// The most specific code wins — AND its own launch descriptor comes with it.
///
/// This is the pairing that can go wrong silently: `select_block` picks the cubin compiled for this
/// card over the portable PTX beside it, and the entry point must then be the CUBIN's. Querying with
/// whatever the stage happens to list first would launch the right module at the wrong symbol, or
/// the right symbol with the wrong block size.
TEST(Rf3mRoundTrip, TheSelectedBlocksOwnLaunchDescriptorIsUsed) {
    PackageBuilder builder = sample_builder();
    builder.block(BlockKind::CudaPtx, "encoder", "", b("portable-ptx"))
        .block(BlockKind::CudaCubin, "encoder", "sm_120", b("cubin-for-this-card"))
        .buffer("features", 8)
        .stage("encoder", Invocation::Once)
        .writes("out", "features")
        .launch(BlockKind::CudaPtx, "", KernelLaunch{"encode_ptx", {128, 1, 1}, {0, 0, 0}, 0})
        .launch(BlockKind::CudaCubin, "sm_120", KernelLaunch{"encode_cubin", {256, 1, 1}, {0, 0, 0}, 0})
        .done()
        .stage("trunk", Invocation::PerQuery)
        .reads("features", "features")
        .done();
    const Package package = Package::read(builder.build().to_bytes());
    const auto composition = package.get_composition();
    const Stage* encoder = composition->get_stage("encoder");

    // On the card the cubin was built for: the cubin, and the cubin's entry point.
    const Block* exact = package.select_block("encoder", "cuda", "sm_120");
    ASSERT_NE(exact, nullptr);
    EXPECT_EQ(exact->kind, BlockKind(BlockKind::CudaCubin));
    const KernelLaunch* for_exact = encoder->get_launch_for(exact->kind.get_name(), exact->target_arch);
    ASSERT_NE(for_exact, nullptr);
    EXPECT_EQ(for_exact->entry_point, "encode_cubin");
    EXPECT_EQ(for_exact->block_size[0], 256u);

    // On any other NVIDIA card: the PTX the driver JITs, and the PTX's entry point.
    const Block* portable = package.select_block("encoder", "cuda", "sm_86");
    ASSERT_NE(portable, nullptr);
    EXPECT_EQ(portable->kind, BlockKind(BlockKind::CudaPtx));
    const KernelLaunch* for_portable =
        encoder->get_launch_for(portable->kind.get_name(), portable->target_arch);
    ASSERT_NE(for_portable, nullptr);
    EXPECT_EQ(for_portable->entry_point, "encode_ptx");
    EXPECT_EQ(for_portable->block_size[0], 128u);
}

TEST(Rf3mRoundTrip, ALaunchDescriptorMayNotDescribeAGraph) {
    PackageBuilder builder = sample_builder();
    builder.buffer("latent", 8)
        .stage("beam_encoder", Invocation::Once)
        .writes("linear_5", "latent")
        // `beam_encoder` is an ONNX graph; ONNX Runtime is asked for its calling convention by name,
        // so a second answer recorded beside it could only ever disagree.
        .launch(BlockKind::Onnx, "", KernelLaunch{"forward", {256, 1, 1}, {0, 0, 0}, 0})
        .done()
        .stage("trunk", Invocation::PerQuery)
        .reads("latent", "latent")
        .done();
    EXPECT_THROW((void)builder.build(), RadFiled3D::nn::Exception);
}

// ── storing and loading a file ────────────────────────────────────────────────────────────────────
//
// Everything above round-trips through `to_bytes`/`read`, which proves the encode/decode pair agrees
// with itself. This proves the FILE path: a package written to disk and read back with no bytes
// passing through the test, which is what a trainer does and what a consumer does.

TEST(Rf3mRoundTrip, APackageSurvivesAFileRoundTrip) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "rfnn_file_round_trip.rf3m";
    std::filesystem::remove(path);

    // Everything the container can carry, so the file path is exercised on a package with a
    // composition, weights and a vendor-specific block rather than on a bare trunk.
    PackageBuilder builder = composed_builder();
    builder.block(BlockKind::CudaPtx, "trunk", "", b("ptx-for-the-trunk"))
        .edit_stage("trunk")
        .weights(b("trunk-parameters"), 4, DType::F32, "trunk.onnx.data")
        .launch(BlockKind::CudaPtx, "", KernelLaunch{"trunk_forward", {64, 1, 1}, {0, 0, 0}, 0});
    const Package original = builder.build();

    original.write_file(path);
    ASSERT_TRUE(std::filesystem::exists(path));
    EXPECT_GT(std::filesystem::file_size(path), kHeaderBytes);

    const Package back = Package::read_file(path);
    EXPECT_EQ(back, original) << "a package must survive a trip through the filesystem unchanged";

    // And the SCAN path over the same file: what a directory listing reads, without touching a
    // payload. It must agree with the full read about everything it reports.
    const Metadata scanned = Package::read_metadata_file(path);
    EXPECT_EQ(scanned.provenance, original.provenance);
    EXPECT_EQ(scanned.io, original.io);
    EXPECT_EQ(scanned.metrics, original.metrics);
    EXPECT_EQ(scanned.composition, original.get_composition());
    ASSERT_EQ(scanned.blocks.size(), original.blocks.size());
    for (std::size_t i = 0; i < scanned.blocks.size(); ++i) {
        EXPECT_EQ(scanned.blocks[i].kind, original.blocks[i].kind);
        EXPECT_EQ(scanned.blocks[i].name, original.blocks[i].name);
        EXPECT_EQ(scanned.blocks[i].target_arch, original.blocks[i].target_arch);
        EXPECT_EQ(scanned.blocks[i].payload_bytes, original.blocks[i].payload.size())
            << "the scan must report the payload it skipped, block " << i;
    }

    std::filesystem::remove(path);
}

TEST(Rf3mRoundTrip, AFileDamagedOnDiskIsRefusedRatherThanRead) {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "rfnn_file_damaged.rf3m";
    std::filesystem::remove(path);
    sample_package().write_file(path);

    // Flip one byte in the middle of the file, the way a bad transfer or a failing disk would. The
    // digest is what turns that into a refusal instead of a model that infers slightly wrong numbers.
    {
        std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
        ASSERT_TRUE(file.is_open());
        const auto middle = static_cast<std::streamoff>(std::filesystem::file_size(path) / 2);
        file.seekg(middle);
        char byte = 0;
        file.read(&byte, 1);
        byte = static_cast<char>(byte ^ 0xff);
        file.seekp(middle);
        file.write(&byte, 1);
    }

    EXPECT_THROW((void)Package::read_file(path), RadFiled3D::nn::Exception);
    std::filesystem::remove(path);
}

TEST(Rf3mRoundTrip, AMissingFileSaysWhichOne) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "rfnn_definitely_not_here.rf3m";
    std::filesystem::remove(path);
    try {
        (void)Package::read_file(path);
        FAIL() << "reading a file that is not there must throw";
    } catch (const RadFiled3D::nn::Exception& e) {
        EXPECT_NE(std::string(e.what()).find("rfnn_definitely_not_here"), std::string::npos) << e.what();
    }
}

// ── stage weights ─────────────────────────────────────────────────────────────────────────────────
//
// One buffer per stage, shared by that stage's IMPLEMENTATIONS. A PTX kernel, an HSACO object and an
// ONNX graph are three ways to compute the same function, so they read one set of numbers; a copy
// per implementation would be three things that can drift apart.

TEST(Rf3mRoundTrip, AStagesWeightsAreOneBufferSharedByItsImplementations) {
    PackageBuilder builder = sample_builder();
    builder.block(BlockKind::CudaPtx, "trunk", "", b("ptx"))
        .block(BlockKind::Hsaco, "trunk", "", b("hsaco"))
        .stage("trunk", Invocation::PerQuery)
        .weights(b("half-precision-weights"), /*elements=*/11, DType::F16, "trunk.onnx.data")
        .done();
    const Package package = Package::read(builder.build().to_bytes());

    // ONE block, whatever the implementation asking for it. Nothing keyed by kind or architecture.
    const auto stored = package.get_weights("trunk");
    EXPECT_EQ(bytes(stored.begin(), stored.end()), b("half-precision-weights"));
    const auto composition = package.get_composition();
    const Stage* trunk = composition->get_stage("trunk");
    EXPECT_EQ(trunk->weights.elements, 11u);
    EXPECT_EQ(trunk->weights.dtype, DType::F16);
    EXPECT_EQ(trunk->weights.onnx_external_file, "trunk.onnx.data");

    // It is a payload, not a way to run the model: skippable by length like any block, and never
    // mistaken for something a stage could be implemented by.
    const auto executable = package.get_executable_block_names();
    EXPECT_NE(std::find(executable.begin(), executable.end(), "trunk"), executable.end());
    for (const auto& block : package.blocks)
        if (block.kind == BlockKind::Weights) EXPECT_FALSE(block.kind.is_executable());
}

TEST(Rf3mRoundTrip, AStageWithNoWeightsHasAnEmptyBufferRatherThanNone) {
    // A stage with no parameters — an activation, a reduction — still HAS the buffer, so a consumer
    // reads its weights the same way whether or not there are any.
    const Package package = sample_package();
    EXPECT_TRUE(package.get_weights("trunk").empty());
    EXPECT_TRUE(package.get_weights("no_such_stage").empty());
    // And nothing is stored for it: a zero-length block and an absent one say the same thing.
    EXPECT_TRUE(package.get_weights_block_names().empty());
}

TEST(Rf3mRoundTrip, WeightsTheStageDeclaresButThePackageDoesNotCarryAreRefused) {
    // Assembled by hand: the builder stores the block and the declaration together, so the only way
    // to reach this state is a producer that wrote the composition itself.
    Package package = sample_builder().build();
    Composition composition;
    composition.stages.push_back(
        Stage{"trunk", Invocation::PerQuery, {}, {}, {}, Weights{7, DType::F32, ""}});
    package.blocks.push_back(
        Block{BlockKind::Composition, std::string(kCompositionBlock), {}, composition.to_bytes()});

    try {
        package.validate();
        FAIL() << "a stage may not declare weights the package does not carry";
    } catch (const RadFiled3D::nn::Exception& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("trunk"), std::string::npos) << what;
        EXPECT_NE(what.find("weights"), std::string::npos) << what;
    }
}

// ── what a composition may not say ────────────────────────────────────────────────────────────────

TEST(Rf3mRoundTrip, AStageNamingNoBlockIsRefused) {
    PackageBuilder builder = sample_builder();
    builder.stage("beam_encoder", Invocation::Once)
        .done()
        .stage("nonexistent_encoder", Invocation::Once)
        .done()
        .stage("trunk", Invocation::PerQuery)
        .done();
    EXPECT_THROW((void)builder.build(), RadFiled3D::nn::Exception);
}

/// THE ordering rule: a buffer holds a value only once its producer has run.
TEST(Rf3mRoundTrip, AStageMayNotReadABufferALaterStageWrites) {
    PackageBuilder builder = sample_builder();
    // The trunk runs last, so nothing it writes can reach the encoder.
    builder.buffer("from_the_future", 4)
        .stage("beam_encoder", Invocation::Once)
        .reads("something", "from_the_future")
        .done()
        .stage("trunk", Invocation::PerQuery)
        .writes("flux", "from_the_future")
        .done();
    try {
        (void)builder.build();
        FAIL() << "reading a buffer a later stage writes must be refused";
    } catch (const RadFiled3D::nn::Exception& e) {
        const std::string what = e.what();
        // The message has to place both ends in the order, or it is not actionable.
        EXPECT_NE(what.find("beam_encoder"), std::string::npos) << what;
        EXPECT_NE(what.find("trunk"), std::string::npos) << what;
        EXPECT_NE(what.find("earlier"), std::string::npos) << what;
    }
}

TEST(Rf3mRoundTrip, AStageMayNotReadABufferItWritesItself) {
    PackageBuilder builder = sample_builder();
    // Same position, not an earlier one: at the moment the stage runs, the buffer holds whatever the
    // previous inference left in it.
    builder.buffer("scratch", 4)
        .stage("beam_encoder", Invocation::Once)
        .writes("linear_5", "scratch")
        .reads("seed", "scratch")
        .done()
        .stage("trunk", Invocation::PerQuery)
        .done();
    EXPECT_THROW((void)builder.build(), RadFiled3D::nn::Exception);
}

TEST(Rf3mRoundTrip, TheTrunkMustRunLastAndPerQuery) {
    PackageBuilder ends_wrong = sample_builder();
    ends_wrong.stage("trunk", Invocation::PerQuery).done().stage("beam_encoder", Invocation::Once).done();
    EXPECT_THROW((void)ends_wrong.build(), RadFiled3D::nn::Exception);

    PackageBuilder runs_once = sample_builder();
    runs_once.stage("beam_encoder", Invocation::Once).done().stage("trunk", Invocation::Once).done();
    EXPECT_THROW((void)runs_once.build(), RadFiled3D::nn::Exception);
}

TEST(Rf3mRoundTrip, ABufferWrittenByTwoStagesIsRefused) {
    PackageBuilder builder = composed_builder();
    // `beam_latent` now has two producers and nothing says which wins.
    builder.edit_stage("encoding_config").writes("also_latent", "beam_latent");
    EXPECT_THROW((void)builder.build(), RadFiled3D::nn::Exception);
}

TEST(Rf3mRoundTrip, ABufferNothingReadsIsRefused) {
    PackageBuilder builder = sample_builder();
    builder.buffer("dead_end", 4)
        .stage("beam_encoder", Invocation::Once)
        .writes("linear_5", "dead_end")
        .done()
        .stage("trunk", Invocation::PerQuery)
        .done();
    // An output the CALLER wants carries no port at all; wiring a producer to nothing is a mistake.
    EXPECT_THROW((void)builder.build(), RadFiled3D::nn::Exception);
}

TEST(Rf3mRoundTrip, AStageThatRunsOnceMayNotConsumeAPerQueryValue) {
    PackageBuilder builder = sample_builder();
    // `post` runs once per field, so it cannot read something computed at every voxel.
    builder.graph("post", b("post-onnx"))
        .buffer("per_voxel", 4)
        .stage("beam_encoder", Invocation::PerQuery)
        .writes("linear_5", "per_voxel")
        .done()
        .stage("post", Invocation::Once)
        .reads("latent", "per_voxel")
        .done()
        .stage("trunk", Invocation::PerQuery)
        .done();
    EXPECT_THROW((void)builder.build(), RadFiled3D::nn::Exception);
}

TEST(Rf3mRoundTrip, AStageMayNotRunTwice) {
    PackageBuilder builder = sample_builder();
    builder.stage("beam_encoder", Invocation::Once)
        .done()
        .stage("beam_encoder", Invocation::Once)
        .done()
        .stage("trunk", Invocation::PerQuery)
        .done();
    EXPECT_THROW((void)builder.build(), RadFiled3D::nn::Exception);
}

/// The layout the format exists for: parse the header by hand, seek past the metadata, and list
/// what a package holds without touching a payload.
///
/// This walks the bytes exactly as an external tool would — no library types beyond the builder
/// that produced the file — so it pins the on-disk contract rather than the API's opinion of it.
TEST(Rf3mRoundTrip, TheHeaderLetsAReaderSkipStraightToTheBlocks) {
    PackageBuilder builder = sample_builder();
    builder.block(BlockKind::CudaPtx, "trunk", "", bytes(4096, 0xab));
    const bytes data = builder.build().to_bytes();

    std::size_t at = 0;
    EXPECT_EQ(std::string(data.begin(), data.begin() + 4), "RF3M");
    at += 4;
    EXPECT_EQ(le32(data, at), 1u) << "file version";
    at += 4;
    at += 32;  // digest

    // One length word, and the whole metadata region is behind us.
    const auto metadata_bytes = static_cast<std::size_t>(le64(data, at));
    at += 8;
    EXPECT_GT(metadata_bytes, 0u);
    at += metadata_bytes;

    const std::uint32_t block_count = le32(data, at);
    at += 4;
    EXPECT_EQ(block_count, 3u) << "trunk, beam_encoder, and the PTX";

    struct Seen {
        std::string kind, name, arch;
        std::size_t payload;
        bool operator==(const Seen&) const = default;
    };
    std::vector<Seen> seen;
    for (std::uint32_t i = 0; i < block_count; ++i) {
        const auto block_bytes = static_cast<std::size_t>(le64(data, at));
        at += 8;
        const std::size_t end = at + block_bytes;
        // Reading the descriptors is cheap; the payload is never touched.
        auto field = [&] {
            const std::uint32_t len = le32(data, at);
            at += 4;
            std::string s(data.begin() + static_cast<std::ptrdiff_t>(at), data.begin() + static_cast<std::ptrdiff_t>(at + len));
            at += len;
            return s;
        };
        const std::string kind = field();
        const std::string name = field();
        const std::string arch = field();
        seen.push_back({kind, name, arch, end - at});
        at = end;  // one seek past a payload we did not want
    }
    EXPECT_EQ(at, data.size()) << "the walk consumed the file exactly";

    const std::vector<Seen> expected = {
        {"onnx", "trunk", "", 16}, {"onnx", "beam_encoder", "", 18}, {"cuda_ptx", "trunk", "", 4096}};
    EXPECT_EQ(seen, expected);

    // The library's own scan agrees with the hand-walk, and reports the same sizes.
    std::vector<Seen> scanned;
    for (const auto& blk : Package::read_metadata(data).blocks)
        scanned.push_back({std::string(blk.kind.get_name()), blk.name, blk.target_arch, static_cast<std::size_t>(blk.payload_bytes)});
    EXPECT_EQ(scanned, expected);
}

// ── integrity ─────────────────────────────────────────────────────────────────────────────────────
//
// The header carries a BLAKE3 digest of everything after it, checked on EVERY read — a full parse
// and a metadata scan alike. It is what lets a consumer say a package is intact rather than hope so,
// and it is the reason a truncated download or a half-written file fails loudly instead of decoding
// into a plausible model.
//
// What it covers is deliberate: the digest spans from the metadata length word to the end of the
// last block. The magic and the version word in front of it are validated directly and produce their
// own diagnoses, so every byte of the file is checked by something — a corrupt magic is
// `BadMagic`, a corrupt version `UnsupportedVersion`, and anything else `DigestMismatch`.

namespace {

/// The same bytes with one flipped, at an offset given from the end so the test does not have to
/// know the header layout.
bytes with_a_flipped_byte(bytes data, std::size_t from_end) {
    data[data.size() - from_end] ^= 0xFF;
    return data;
}

}  // namespace

TEST(Integrity, ATamperedPayloadIsRefused) {
    const bytes good = sample_package().to_bytes();
    EXPECT_NO_THROW((void)Package::read(good));

    // A byte in the middle of a block payload: the kind of change that would otherwise decode
    // perfectly well into a model that is quietly not the one that was trained.
    EXPECT_THROW((void)Package::read(with_a_flipped_byte(good, 16)), RadFiled3D::nn::Exception);
    try {
        (void)Package::read(with_a_flipped_byte(good, 16));
    } catch (const RadFiled3D::nn::Exception& err) {
        EXPECT_EQ(err.get_kind(), RadFiled3D::nn::ErrorKind::DigestMismatch) << err.what();
    }
}

TEST(Integrity, ATamperedMetadataRegionIsRefused) {
    const bytes good = sample_package().to_bytes();
    // Far enough in to land in the descriptors rather than a payload.
    EXPECT_THROW((void)Package::read(with_a_flipped_byte(good, good.size() - 80)),
                 RadFiled3D::nn::Exception);
}

TEST(Integrity, TheScanChecksTheDigestToo) {
    // `read_metadata` touches no payload, but it must still refuse a package it cannot vouch for —
    // a listing that reported a corrupt file as if it were fine would be worse than no listing.
    const bytes good = sample_package().to_bytes();
    EXPECT_NO_THROW((void)Package::read_metadata(good));
    EXPECT_THROW((void)Package::read_metadata(with_a_flipped_byte(good, 16)), RadFiled3D::nn::Exception);
}

TEST(Integrity, ACorruptMagicAndACorruptVersionEachSayWhatTheyAre) {
    bytes data = sample_package().to_bytes();
    bytes bad_magic = data;
    bad_magic[1] = 'X';
    try {
        (void)Package::read(bad_magic);
        ADD_FAILURE() << "a package with the wrong magic was accepted";
    } catch (const RadFiled3D::nn::Exception& err) {
        EXPECT_EQ(err.get_kind(), RadFiled3D::nn::ErrorKind::BadMagic) << err.what();
    }

    bytes bad_version = data;
    bad_version[4] = 99;
    try {
        (void)Package::read(bad_version);
        ADD_FAILURE() << "a package with an unknown version was accepted";
    } catch (const RadFiled3D::nn::Exception& err) {
        // Not a digest failure: the version is checked before the digest, and "I cannot read this
        // layout" is a different problem from "these bytes are damaged".
        EXPECT_EQ(err.get_kind(), RadFiled3D::nn::ErrorKind::UnsupportedVersion) << err.what();
    }
}

TEST(Integrity, ATruncatedPackageIsRefused) {
    const bytes good = sample_package().to_bytes();
    for (const std::size_t keep : {std::size_t{0}, std::size_t{8}, good.size() / 2, good.size() - 1})
        EXPECT_THROW((void)Package::read(bytes(good.begin(), good.begin() + static_cast<std::ptrdiff_t>(keep))),
                     RadFiled3D::nn::Exception)
            << "truncated to " << keep << " bytes";
}

TEST(Integrity, TheDigestChangesWithThePayload) {
    // Two packages differing only in a graph must not share a digest; otherwise the check would be
    // vacuous for exactly the substitution it exists to catch.
    PackageBuilder other = sample_builder();
    other.graph("trunk", b("a-different-trunk"));
    const bytes a = sample_package().to_bytes();
    const bytes c = other.build().to_bytes();
    const bytes digest_a(a.begin() + 8, a.begin() + 40);
    const bytes digest_c(c.begin() + 8, c.begin() + 40);
    EXPECT_NE(digest_a, digest_c);
}
