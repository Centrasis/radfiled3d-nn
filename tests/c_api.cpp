// The C ABI.
//
// This is the surface an engine plugin actually calls, and the one place where a mistake is
// undefined behaviour in someone else's process rather than a failed assertion. The functions are
// exercised exactly as C would call them — raw pointers in, status codes out, handles freed
// explicitly — and through the SHARED library, so a symbol missing from its export list fails here.
#include <RadFiled3D/nn/c_api.h>
#include <RadFiled3D/nn/c_api.hpp>
#include <RadFiled3D/nn/deploy.hpp>
#include <RadFiled3D/nn/version.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace {

std::filesystem::path write_sample(const char* file_name) {
    using namespace RadFiled3D::nn::deploy;
    const auto path = std::filesystem::temp_directory_path() / file_name;
    PackageBuilder builder;
    builder.provenance("xray-scatter-v3", "radfield3d-nn 2.1", "physics")
        .field_dimensions_m({1.f, 1.f, 1.f})
        .input("position", Semantic::Position, {3}).unit("m").done()
        .input("beam_direction", Semantic::BeamDirection, {2}).unit("rad").done()
        .output("flux", Semantic::Flux, {1}).normalizer(LogScale{1e-12, 30.0}).done()
        .output("direction_distribution", Semantic::from_name("direction_distribution"), {16, 32}).done()
        .graph("trunk", {'<', 'o', 'n', 'n', 'x', '>'});
    builder.build().write_file(path);
    return path;
}

}  // namespace

TEST(CApi, VersionIsAStaticString) {
    EXPECT_STREQ(rfnn_version(), RFNN_VERSION);
}

TEST(CApi, APackageRoundTripsThroughTheCAbi) {
    const auto path = write_sample("capi_round_trip.rf3m");
    rfnn_metadata* handle = nullptr;
    ASSERT_EQ(rfnn_metadata_read(path.string().c_str(), &handle), RFNN_OK);
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(rfnn_metadata_tensor_count(handle), 4u);

    // Names and roles come back in declaration order.
    std::vector<std::string> inputs, outputs;
    for (uint32_t i = 0; i < rfnn_metadata_tensor_count(handle); ++i) {
        const char* name = rfnn_metadata_tensor_name(handle, i);
        ASSERT_NE(name, nullptr);
        (rfnn_metadata_tensor_is_output(handle, i) == 1 ? outputs : inputs).emplace_back(name);
    }
    EXPECT_EQ(inputs, (std::vector<std::string>{"position", "beam_direction"}));
    EXPECT_EQ(outputs, (std::vector<std::string>{"flux", "direction_distribution"}));

    // Out-of-range indices are reported, never read out of bounds.
    EXPECT_EQ(rfnn_metadata_tensor_name(handle, 4), nullptr);
    EXPECT_EQ(rfnn_metadata_tensor_is_output(handle, 4), -1);
    rfnn_metadata_free(handle);

    // The RAII wrapper is a projection of the same calls.
    RadFiled3D::nn::c::Metadata md(path.string());
    EXPECT_EQ(md.output_names(), outputs);
    std::filesystem::remove(path);
}

TEST(CApi, AMissingFileReportsAMessageRatherThanABareCode) {
    rfnn_metadata* handle = nullptr;
    EXPECT_EQ(rfnn_metadata_read("/nonexistent/definitely-not-a-package.rf3m", &handle), RFNN_IO);
    EXPECT_EQ(handle, nullptr) << "a failed read must not hand back a handle";
    ASSERT_NE(rfnn_last_error(), nullptr);
    EXPECT_NE(std::strstr(rfnn_last_error(), "definitely-not-a-package.rf3m"), nullptr);
    EXPECT_THROW(RadFiled3D::nn::c::Metadata("/nonexistent/x.rf3m"), RadFiled3D::nn::c::Exception);
}

TEST(CApi, ACorruptPackageIsReportedAsABadPackage) {
    const auto path = write_sample("capi_corrupt.rf3m");
    {
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(-1, std::ios::end);
        const char last = static_cast<char>(0xff);
        f.write(&last, 1);
    }
    rfnn_metadata* handle = nullptr;
    EXPECT_EQ(rfnn_metadata_read(path.string().c_str(), &handle), RFNN_BAD_PACKAGE);
    EXPECT_EQ(handle, nullptr);
    std::filesystem::remove(path);
}

TEST(CApi, NullArgumentsAreRefusedInsteadOfDereferenced) {
    rfnn_metadata* handle = nullptr;
    EXPECT_EQ(rfnn_metadata_read(nullptr, &handle), RFNN_INVALID_ARGUMENT);
    EXPECT_EQ(rfnn_metadata_read("whatever.rf3m", nullptr), RFNN_INVALID_ARGUMENT);
    // Accessors tolerate a null handle, and freeing null is a no-op — both are things a C caller
    // does after an error it did not check.
    EXPECT_EQ(rfnn_metadata_tensor_count(nullptr), 0u);
    EXPECT_EQ(rfnn_metadata_tensor_name(nullptr, 0), nullptr);
    EXPECT_EQ(rfnn_metadata_tensor_is_output(nullptr, 0), -1);
    rfnn_metadata_free(nullptr);
}

// ── the inference surface ────────────────────────────────────────────────────────────────────────
//
// Driven the way C drives it: opaque handles, status codes, explicit frees. This is the surface an
// engine plugin actually calls, so a mistake here is undefined behaviour in someone else's process.

TEST(CApi, TensorMetadataCrossesTheBoundary) {
    const auto path = write_sample("capi_tensor_metadata.rf3m");
    rfnn_metadata* handle = nullptr;
    ASSERT_EQ(rfnn_metadata_read(path.string().c_str(), &handle), RFNN_OK);

    // position: [3], metres, no range recorded.
    EXPECT_STREQ(rfnn_metadata_tensor_semantic(handle, 0), "position");
    EXPECT_STREQ(rfnn_metadata_tensor_unit(handle, 0), "m");
    EXPECT_EQ(rfnn_metadata_tensor_rank(handle, 0), 1);
    uint32_t shape[4] = {0, 0, 0, 0};
    EXPECT_EQ(rfnn_metadata_tensor_shape(handle, 0, shape, 4), 1);
    EXPECT_EQ(shape[0], 3u);
    EXPECT_EQ(rfnn_metadata_tensor_range(handle, 0, nullptr, nullptr), 0) << "no range was recorded";

    // flux carries a log-scale normalizer; the caller never has to know its parameters.
    EXPECT_STREQ(rfnn_metadata_tensor_normalizer(handle, 2), "log_scale");

    // A semantic this build has never heard of still comes back as itself — that is the point of
    // the descriptor design, and it has to survive the C boundary too.
    EXPECT_STREQ(rfnn_metadata_tensor_semantic(handle, 3), "direction_distribution");
    EXPECT_EQ(rfnn_metadata_tensor_rank(handle, 3), 2);
    EXPECT_EQ(rfnn_metadata_tensor_shape(handle, 3, shape, 4), 2);
    EXPECT_EQ(shape[0], 16u);
    EXPECT_EQ(shape[1], 32u);

    // A shape query with no buffer reports how big one would need to be.
    EXPECT_EQ(rfnn_metadata_tensor_shape(handle, 3, nullptr, 0), 2);
    // ... and a short buffer is filled, not overrun.
    uint32_t one = 0;
    EXPECT_EQ(rfnn_metadata_tensor_shape(handle, 3, &one, 1), 1);
    EXPECT_EQ(one, 16u);

    // Out of range is reported, never read out of bounds.
    EXPECT_EQ(rfnn_metadata_tensor_semantic(handle, 99), nullptr);
    EXPECT_EQ(rfnn_metadata_tensor_rank(handle, 99), -1);
    EXPECT_EQ(rfnn_metadata_tensor_range(handle, 99, nullptr, nullptr), -1);
    rfnn_metadata_free(handle);
    std::filesystem::remove(path);
}

TEST(CApi, HostMemoryCrossesAsAnOpaqueHandle) {
    std::vector<float> values(256, 1.5f);
    rfnn_memory* memory = nullptr;
    ASSERT_EQ(rfnn_memory_from_host(values.data(), values.size() * sizeof(float), &memory), RFNN_OK);
    ASSERT_NE(memory, nullptr);
    EXPECT_EQ(rfnn_memory_size_bytes(memory), values.size() * sizeof(float));
    EXPECT_EQ(rfnn_memory_domain(memory), RFNN_DOMAIN_HOST);
    rfnn_memory_free(memory);

    // Null-tolerant, like every other accessor here.
    EXPECT_EQ(rfnn_memory_size_bytes(nullptr), 0u);
    EXPECT_EQ(rfnn_memory_domain(nullptr), -1);
    rfnn_memory_free(nullptr);

    // A null pointer with a non-zero size is a caller bug, not an empty buffer.
    rfnn_memory* bad = nullptr;
    EXPECT_EQ(rfnn_memory_from_host(nullptr, 16, &bad), RFNN_INVALID_ARGUMENT);
    EXPECT_EQ(bad, nullptr);
    EXPECT_EQ(rfnn_memory_from_host(values.data(), 4, nullptr), RFNN_INVALID_ARGUMENT);
}

TEST(CApi, AD3D11ResourceCrossesWithoutBeingImported) {
    // The Unity shape: the engine owns the D3D11 device, so its resource is not imported — it is
    // DESCRIBED, and stays in the D3D11 domain while the engine goes on using it.
    void* const resource = reinterpret_cast<void*>(std::uintptr_t{0xd3d11});
    rfnn_memory* memory = nullptr;
    ASSERT_EQ(rfnn_memory_from_d3d11_resource(resource, /*shared_handle=*/nullptr, RFNN_D3D11_NT,
                                              4096, 0, 4096, nullptr, &memory),
              RFNN_OK);
    ASSERT_NE(memory, nullptr);
    EXPECT_EQ(rfnn_memory_domain(memory), RFNN_DOMAIN_D3D11);
    EXPECT_EQ(rfnn_memory_size_bytes(memory), 4096u);
    rfnn_memory_free(memory);

    // With a shared handle it describes the same resource AND carries what a compute backend needs
    // to import it. Both handle kinds are accepted; which one it is decides the CUDA import type.
    for (const rfnn_d3d11_kind kind : {RFNN_D3D11_NT, RFNN_D3D11_KMT}) {
        rfnn_memory* shareable = nullptr;
        ASSERT_EQ(rfnn_memory_from_d3d11_resource(resource,
                                                  reinterpret_cast<void*>(std::uintptr_t{0x5ade}),
                                                  kind, 4096, 256, 1024, nullptr, &shareable),
                  RFNN_OK);
        ASSERT_NE(shareable, nullptr);
        EXPECT_EQ(rfnn_memory_domain(shareable), RFNN_DOMAIN_D3D11);
        // The region, not the whole allocation: a renderer suballocates.
        EXPECT_EQ(rfnn_memory_size_bytes(shareable), 1024u);
        rfnn_memory_free(shareable);
    }

    // A null resource is a caller bug: there is nothing to denote.
    rfnn_memory* bad = nullptr;
    EXPECT_EQ(rfnn_memory_from_d3d11_resource(nullptr, nullptr, RFNN_D3D11_NT, 16, 0, 16, nullptr, &bad),
              RFNN_INVALID_ARGUMENT);
    EXPECT_EQ(bad, nullptr);
    EXPECT_EQ(rfnn_memory_from_d3d11_resource(resource, nullptr, RFNN_D3D11_NT, 16, 0, 16, nullptr, nullptr),
              RFNN_INVALID_ARGUMENT);
}

TEST(CApi, AnUnknownBackendNameIsRefusedAtTheBoundary) {
    // The vocabulary is the C++ one; this ABI forwards a name and re-decides nothing, so a typo
    // must fail here rather than quietly selecting a default.
    rfnn_session* session = nullptr;
    const auto path = write_sample("capi_backend_name.rf3m");
    EXPECT_EQ(rfnn_session_load(path.string().c_str(), "gpu", -1, &session), RFNN_INVALID_ARGUMENT);
    EXPECT_EQ(session, nullptr);
    ASSERT_NE(rfnn_last_error(), nullptr);
    EXPECT_NE(std::string(rfnn_last_error()).find("gpu"), std::string::npos);

    std::vector<float> values(4, 0.f);
    rfnn_memory* memory = nullptr;
    EXPECT_EQ(rfnn_memory_import_vulkan_fd(-1, 16, 0, 16, nullptr, 0, "nonsense", &memory),
              RFNN_INVALID_ARGUMENT);
    EXPECT_EQ(memory, nullptr);
    std::filesystem::remove(path);
}

TEST(CApi, NullArgumentsAcrossTheInferenceSurface) {
    rfnn_session* session = nullptr;
    EXPECT_EQ(rfnn_session_load(nullptr, "cpu", -1, &session), RFNN_INVALID_ARGUMENT);
    EXPECT_EQ(rfnn_session_load("x.rf3m", nullptr, -1, &session), RFNN_INVALID_ARGUMENT);
    EXPECT_EQ(rfnn_session_load("x.rf3m", "cpu", -1, nullptr), RFNN_INVALID_ARGUMENT);

    // Everything tolerates a null session: a C caller reaches these after an error it did not check.
    EXPECT_EQ(rfnn_session_set_voxel_grid(nullptr, 1, 1, 1), RFNN_INVALID_ARGUMENT);
    EXPECT_EQ(rfnn_session_bind_input(nullptr, "position", nullptr), RFNN_INVALID_ARGUMENT);
    EXPECT_EQ(rfnn_session_bind_output(nullptr, "flux", nullptr), RFNN_INVALID_ARGUMENT);
    EXPECT_EQ(rfnn_session_infer(nullptr), RFNN_INVALID_ARGUMENT);
    EXPECT_EQ(rfnn_session_backend(nullptr), nullptr);
    rfnn_session_free(nullptr);
}

/// The whole protocol through C, against a real model. Skips without one, and without ONNX Runtime.
TEST(CApi, TheInferenceProtocolRunsThroughC) {
    const char* env = std::getenv("RFNN_TEST_MODEL");
    const std::string model_path =
        env ? env : "/mnt/data/models_nn/logs/pbrf-ds03-3779-noroi/pbrf-ds03-3779-noroi/models/PBRFNet.rf3m";
    if (!std::filesystem::exists(model_path)) GTEST_SKIP() << "no model at " << model_path;

    rfnn_session* session = nullptr;
    const rfnn_status loaded = rfnn_session_load(model_path.c_str(), "cpu", -1, &session);
    if (loaded == RFNN_FEATURE_DISABLED) GTEST_SKIP() << "built without ONNX Runtime";
    ASSERT_EQ(loaded, RFNN_OK) << rfnn_last_error();
    EXPECT_STREQ(rfnn_session_backend(session), "cpu");

    constexpr uint32_t kSide = 4;
    constexpr std::size_t kQueries = kSide * kSide * kSide;
    ASSERT_EQ(rfnn_session_set_voxel_grid(session, kSide, kSide, kSide), RFNN_OK);

    // Running with nothing bound must fail, and name every missing binding rather than the first.
    EXPECT_EQ(rfnn_session_infer(session), RFNN_INVALID_ARGUMENT);
    const std::string missing = rfnn_last_error();
    EXPECT_NE(missing.find("position"), std::string::npos) << missing;
    EXPECT_NE(missing.find("flux"), std::string::npos) << missing;

    std::vector<float> position(kQueries * 3, 0.5f), latent(kQueries * 192, 0.01f),
        region(14, 0.f), flux(kQueries, 0.f), spectrum(kQueries * 32, 0.f);
    auto wrap = [](std::vector<float>& v) {
        rfnn_memory* m = nullptr;
        EXPECT_EQ(rfnn_memory_from_host(v.data(), v.size() * sizeof(float), &m), RFNN_OK);
        return m;
    };
    rfnn_memory* m_position = wrap(position);
    rfnn_memory* m_latent = wrap(latent);
    rfnn_memory* m_region = wrap(region);
    rfnn_memory* m_flux = wrap(flux);
    rfnn_memory* m_spectrum = wrap(spectrum);

    ASSERT_EQ(rfnn_session_bind_input(session, "position", m_position), RFNN_OK) << rfnn_last_error();
    ASSERT_EQ(rfnn_session_bind_input(session, "latent", m_latent), RFNN_OK);
    ASSERT_EQ(rfnn_session_bind_input(session, "region_state", m_region), RFNN_OK);
    ASSERT_EQ(rfnn_session_bind_output(session, "flux", m_flux), RFNN_OK);
    ASSERT_EQ(rfnn_session_bind_output(session, "spectrum", m_spectrum), RFNN_OK);

    // Freeing the handles now is legal: the session took its own references. If it did not, the run
    // below would read freed memory — which is exactly the mistake this checks for.
    for (rfnn_memory* m : {m_position, m_latent, m_region, m_flux, m_spectrum}) rfnn_memory_free(m);

    ASSERT_EQ(rfnn_session_infer(session), RFNN_OK) << rfnn_last_error();
    EXPECT_TRUE(std::all_of(flux.begin(), flux.end(), [](float v) { return std::isfinite(v); }));
    EXPECT_NE(std::count_if(flux.begin(), flux.end(), [](float v) { return v != 0.f; }), 0)
        << "the output buffer is still all zeros";

    // A binding whose name the graph does not have is reported, not ignored.
    rfnn_memory* stray = wrap(flux);
    EXPECT_EQ(rfnn_session_bind_output(session, "not_a_tensor", stray), RFNN_NOT_FOUND);
    rfnn_memory_free(stray);

    rfnn_session_free(session);
}

/// The RAII wrapper is a projection of the same calls — no second implementation (R-X3).
TEST(CApi, TheRaiiWrapperDrivesTheSameProtocol) {
    const char* env = std::getenv("RFNN_TEST_MODEL");
    const std::string model_path =
        env ? env : "/mnt/data/models_nn/logs/pbrf-ds03-3779-noroi/pbrf-ds03-3779-noroi/models/PBRFNet.rf3m";
    if (!std::filesystem::exists(model_path)) GTEST_SKIP() << "no model at " << model_path;

    constexpr std::size_t kQueries = 8;
    std::vector<float> position(kQueries * 3, 0.5f), latent(kQueries * 192, 0.01f), region(14, 0.f),
        flux(kQueries, 0.f), spectrum(kQueries * 32, 0.f);
    try {
        RadFiled3D::nn::c::Session session(model_path, "cpu");
        EXPECT_EQ(session.backend(), "cpu");
        session.set_voxel_grid(2, 2, 2);
        session.bind_input("position", RadFiled3D::nn::c::Memory::host(position.data(), position.size() * 4));
        session.bind_input("latent", RadFiled3D::nn::c::Memory::host(latent.data(), latent.size() * 4));
        session.bind_input("region_state", RadFiled3D::nn::c::Memory::host(region.data(), region.size() * 4));
        session.bind_output("flux", RadFiled3D::nn::c::Memory::host(flux.data(), flux.size() * 4));
        session.bind_output("spectrum", RadFiled3D::nn::c::Memory::host(spectrum.data(), spectrum.size() * 4));
        // Every Memory above is already destroyed — the session holds what it needs.
        session.infer();
    } catch (const RadFiled3D::nn::c::Exception& err) {
        if (err.status() == RFNN_FEATURE_DISABLED) GTEST_SKIP() << "built without ONNX Runtime";
        throw;
    }
    EXPECT_NE(std::count_if(flux.begin(), flux.end(), [](float v) { return v != 0.f; }), 0);
}
