// The RadFiled3D boundary.
//
// Deliberately thin: the point is to prove that a field of this library IS a RadFiled3D field —
// it goes through `FieldStore::store` and comes back through `FieldStore::load` — and that the
// metadata conversion never carries a struct image. Not to re-test RadFiled3D itself.
#include <RadFiled3D/nn.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>

using namespace RadFiled3D::nn;

TEST(RadFiled3DBoundary, AFieldHasExactlyTheRequestedGeometry) {
    const CartesianFieldGeometry geometry = CartesianFieldGeometry::cubic(64, 1.f);
    EXPECT_EQ(geometry.get_voxel_dimensions_m(), (std::array<float, 3>{1.f / 64, 1.f / 64, 1.f / 64}));
    EXPECT_EQ(geometry.get_voxel_count(), 64u * 64 * 64);

    auto field = allocate_host_field(geometry);
    // The geometry that comes back is the one RadFiled3D itself derived, not the one we asked for:
    // `allocate_host_field` refuses the field if the two disagree.
    EXPECT_EQ(get_geometry_of(*field), geometry);
}

/// RadFiled3D derives its voxel counts as `trunc((box + epsilon) / voxel_size)`, and that epsilon
/// is what keeps an awkward resolution from losing a voxel to float truncation — 1.0 m in 3 voxels
/// gives a voxel edge of 0.33333334 m, whose quotient is 2.9999999 before the nudge. The counts
/// therefore agree in practice; the check exists so that a case where they do not is an error
/// rather than a field whose every voxel index is off by one.
TEST(RadFiled3DBoundary, AnAcceptedFieldAlwaysHasExactlyTheRequestedGeometry) {
    for (const std::uint32_t resolution : {3u, 7u, 33u, 64u, 100u, 127u}) {
        const CartesianFieldGeometry geometry = CartesianFieldGeometry::cubic(resolution, 1.f);
        try {
            EXPECT_EQ(get_geometry_of(*allocate_host_field(geometry)), geometry) << "resolution " << resolution;
        } catch (const Exception& err) {
            EXPECT_EQ(err.get_kind(), ErrorKind::InvalidArgument) << err.what();
        }
    }
}

TEST(RadFiled3DBoundary, ADegenerateGeometryIsRefusedBeforeItReachesRadFiled3D) {
    EXPECT_THROW((void)CartesianFieldGeometry::make({0, 1, 1}, {1.f, 1.f, 1.f}), Exception);
    EXPECT_THROW((void)CartesianFieldGeometry::make({1, 1, 1}, {0.f, 1.f, 1.f}), Exception);
    EXPECT_THROW((void)CartesianFieldGeometry::make({1, 1, 1}, {1.f, std::nanf(""), 1.f}), Exception);
}

/// The full round trip the module exists to make possible: a GPU field is filled, downloaded to a
/// host `CartesianRadiationField`, and written as `.rf3` by RadFiled3D's own `FieldStore`.
///
/// That last step is the proof that "IS a RadFiled3D field" is literally true — `FieldStore::store`
/// takes a `shared_ptr<IRadiationField>`, so a type that were merely *shaped like* a field could
/// not be passed to it at all.
TEST(RadFiled3DBoundary, AGpuFieldDownloadsAndStoresAsRf3) {
    const CartesianFieldGeometry geometry = CartesianFieldGeometry::cubic(8, 1.f);
    auto gpu = allocate_gpu_field(geometry);
    EXPECT_EQ(gpu->get_voxel_counts(), glm::uvec3(8, 8, 8));
    EXPECT_EQ(gpu->get_geometry(), geometry);

    // The GPU mirror a graphics backend would attach. It crosses as a MemoryRef, so the field never
    // learns which API owns the buffer; null means "none attached".
    EXPECT_EQ(gpu->get_device_memory(std::string(kPredictionChannel), std::string(kFluxLayer)), nullptr);
    auto mirror = std::make_shared<memory::cuda::MemoryRef>(0xdead'beefu, 8 * 8 * 8 * sizeof(float));
    gpu->set_device_memory(std::string(kPredictionChannel), std::string(kFluxLayer), mirror);
    auto attached = gpu->get_device_memory(std::string(kPredictionChannel), std::string(kFluxLayer));
    ASSERT_NE(attached, nullptr);
    EXPECT_EQ(attached->get_domain(), memory::Domain::Cuda);
    EXPECT_EQ(attached->get_size_bytes(), 8u * 8 * 8 * sizeof(float));
    EXPECT_EQ(attached->get_element_count(sizeof(float)), 8u * 8 * 8);
    // The concrete type is reachable when a backend needs the native handle, and only then.
    EXPECT_EQ(std::static_pointer_cast<memory::cuda::MemoryRef>(attached)->get_device_ptr(), 0xdead'beefu);

    // A real IRadiationField: the base-class API works unchanged.
    std::shared_ptr<RadFiled3D::IRadiationField> as_base = gpu;
    EXPECT_TRUE(as_base->has_channel(std::string(kPredictionChannel)));

    auto host = gpu->to_host_field();
    EXPECT_EQ(get_geometry_of(*host), geometry);
    auto channel = host->get_channel(std::string(kPredictionChannel));
    channel->add_layer<float>(std::string(kFluxLayer), 0.f, "1/cm^2/s");
    float* voxels = channel->get_layer<float>(std::string(kFluxLayer));
    for (std::size_t i = 0; i < channel->get_voxel_count(); ++i) voxels[i] = static_cast<float>(i);
    EXPECT_EQ(voxels[511], 511.f);

    const auto path = std::filesystem::temp_directory_path() / "rfnn_gpu_round_trip.rf3";
    store_host_field(host, path, StoreProvenance{"radfiled3d-nn", kVersion, "https://github.com/Centrasis/radfiled3d-nn", ""});
    // Load it back through FieldStore: a file RadFiled3D can re-read is the only real proof that
    // what was stored was a genuine field and not merely bytes.
    auto reloaded = load_host_field(path);
    EXPECT_EQ(reloaded->get_voxel_counts(), glm::uvec3(8, 8, 8));
    const float* values = reloaded->get_channel(std::string(kPredictionChannel))->get_layer<float>(std::string(kFluxLayer));
    EXPECT_EQ(values[0], 0.f);
    EXPECT_EQ(values[511], 511.f);
    std::filesystem::remove(path);
}

TEST(RadFiled3DBoundary, AMissingRf3IsReportedAsIo) {
    try {
        (void)load_host_field("/nonexistent/definitely-not-a-field.rf3");
        FAIL();
    } catch (const Exception& err) {
        EXPECT_EQ(err.get_kind(), ErrorKind::Io);
        EXPECT_NE(std::string(err.what()).find("definitely-not-a-field.rf3"), std::string::npos);
    }
}

/// Every backend answers its availability probe, and the answer matches what this build was
/// compiled with. The probes exist unconditionally, so a disabled backend reports `false` rather
/// than failing to link — which is what lets the API keep its full surface.
TEST(RadFiled3DBoundary, EveryBackendReportsItsAvailability) {
    bool onnx_on = false, cuda_on = false, vulkan_on = false, dx11_on = false, dx12_on = false;
#ifdef RFNN_TEST_WITH_ONNX
    onnx_on = true;
#endif
#ifdef RFNN_TEST_WITH_CUDA
    cuda_on = true;
#endif
#ifdef RFNN_TEST_WITH_VULKAN
    vulkan_on = true;
#endif
#ifdef RFNN_TEST_WITH_DX11
    dx11_on = true;
#endif
#ifdef RFNN_TEST_WITH_DX12
    dx12_on = true;
#endif
    EXPECT_EQ(onnx::available(), onnx_on);
    EXPECT_EQ(cuda::available(), cuda_on);
    EXPECT_EQ(vulkan::available(), vulkan_on);
    EXPECT_EQ(dx11::available(), dx11_on);
    EXPECT_EQ(dx12::available(), dx12_on);
    EXPECT_EQ(is_available(Backend::Cpu), onnx_on);
    // The ONNX Runtime CMake fetched must be genuinely linked, not merely downloaded. Reading its
    // version calls into the library, so this fails if the fetch, the link or the RPATH is wrong.
    if (onnx_on) {
        ASSERT_TRUE(onnx::version().has_value());
        EXPECT_TRUE(std::isdigit(static_cast<unsigned char>(onnx::version()->front()))) << *onnx::version();
    } else {
        EXPECT_FALSE(onnx::version().has_value());
    }
}

/// A graphics backend that was compiled out reports an error naming the feature, never a silent
/// fallback to a host copy (R-G3).
TEST(RadFiled3DBoundary, ACompiledOutGraphicsBackendNamesItsFeature) {
    memory::vk::ExternalMemory vulkan_interop;
    EXPECT_EQ(vulkan_interop.get_domain(), memory::Domain::Vulkan);
    EXPECT_EQ(vulkan_interop.supports(Backend::Cuda), vulkan::available() && cuda::available());
    // The CPU provider can never share graphics memory, whatever is compiled in.
    EXPECT_FALSE(vulkan_interop.supports(Backend::Cpu));

    memory::ExternalBuffer buffer;
    buffer.handle = memory::OpaqueFd{-1};   // never a valid descriptor
    buffer.size_bytes = 1024;
    buffer.region_bytes = 1024;
    try {
        (void)vulkan_interop.import_buffer(buffer, Backend::Cuda);
        FAIL() << "an invalid descriptor must not import";
    } catch (const Exception& err) {
        if (vulkan::available() && cuda::available()) {
            // Both halves are compiled in, so the import really runs and rejects the descriptor.
            EXPECT_EQ(err.get_kind(), ErrorKind::InvalidArgument) << err.what();
        } else {
            // Compiled out: the API still exists and names the option that would enable it (R-G3).
            EXPECT_EQ(err.get_kind(), ErrorKind::FeatureDisabled) << err.what();
            EXPECT_NE(std::string(err.what()).find("RFNN_WITH_"), std::string::npos) << err.what();
        }
    }
    // An impossible pairing is a different error from a compiled-out one.
    EXPECT_THROW(
        {
            try {
                (void)vulkan_interop.import_buffer(buffer, Backend::DirectMl);
            } catch (const Exception& err) {
                EXPECT_EQ(err.get_kind(), ErrorKind::UnsupportedInterop);
                throw;
            }
        },
        Exception);
}

TEST(RadFiled3DBoundary, TheBackendVocabularyIsDefinedOnce) {
    for (const Backend b : kBackends) EXPECT_EQ(backend_from_name(to_string(b)), b);
    EXPECT_THROW((void)backend_from_name("gpu"), Exception);
    // `trunk` here is one byte — not an ONNX graph. Without a runtime, load() cannot get far enough
    // to notice and names the missing option; with one, it gets as far as the parser and says the
    // graph is bad. Both are honest; neither silently succeeds.
    deploy::PackageBuilder builder;
    builder.input("position", deploy::Semantic::Position, {3}).done()
        .output("flux", deploy::Semantic::Flux, {1}).done()
        .graph("trunk", {1});
    const deploy::Package package = builder.build();
    if (!onnx::available()) {
        try {
            (void)load(package, Backend::Cpu);
            FAIL() << "without ONNX Runtime there is nothing to run on";
        } catch (const Exception& err) {
            EXPECT_EQ(err.get_kind(), ErrorKind::FeatureDisabled);
            EXPECT_EQ(err.get_feature(), "onnx");
        }
    } else {
        EXPECT_ANY_THROW((void)load(package, Backend::Cpu)) << "a one-byte `trunk` is not a graph";
    }
}

/// The point of the typed encoding: an RF3M package carries the RadFiled3D provenance of the
/// training data, and RadFiled3D::nn::deploy reads it back with no RadFiled3D linked at all (R-B4).
TEST(RadFiled3DBoundary, Rf3MetadataEmbedsInAnRf3mPackage) {
    using namespace RadFiled3D::Storage;
    auto metadata = std::make_shared<V1::RadiationFieldMetadata>(
        FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
            1'000'000, "patient_phantom", "G4EmStandard",
            FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(glm::vec3(0, 0, 1), glm::vec3(0, 0, -1),
                                                                                150'000.f, "tube-7")),
        FiledTypes::V1::RadiationFieldMetadataHeader::Software("radfiled3d-nn", "0.1", "repo", "abc"));
    metadata->add_dynamic_metadata<float>("beam_angle_deg", 12.5f);

    // The boundary converts RadFiled3D's packed POD into the container's typed, explicitly encoded
    // form. Nothing durable ever carries a struct image.
    const deploy::Rf3Metadata carried = to_package_metadata(*metadata);
    EXPECT_EQ(carried.simulation.primary_particle_count, 1'000'000u);
    EXPECT_EQ(carried.simulation.physics_list, "G4EmStandard");
    EXPECT_EQ(carried.simulation.tube.tube_id, "tube-7");
    EXPECT_EQ(carried.software.name, "radfiled3d-nn");
    EXPECT_FALSE(carried.dynamic.empty()) << "the dynamic layer travels as RadFiled3D's own bytes";

    deploy::PackageBuilder builder;
    builder.field_dimensions_m({1.f, 1.f, 1.f})
        .input("position", deploy::Semantic::Position, {3}).done()
        .output("flux", deploy::Semantic::Flux, {1}).done()
        .graph("trunk", {1})
        .rf3_metadata(carried);
    const deploy::Package package = builder.build();
    const deploy::Package decoded = deploy::Package::read(package.to_bytes());
    EXPECT_EQ(decoded, package);

    // And it converts back into a real RadFiled3D metadata object, dynamic layers included, so a
    // `.rf3` written from an inference result can carry the dataset's provenance.
    ASSERT_TRUE(decoded.rf3_metadata.has_value());
    auto rebuilt = from_package_metadata(*decoded.rf3_metadata);
    EXPECT_EQ(rebuilt->get_header().simulation.tube.max_energy_eV, 150'000.f);
    EXPECT_EQ(std::string(rebuilt->get_header().simulation.physics_list), "G4EmStandard");
    EXPECT_EQ(rebuilt->get_dynamic_metadata<RadFiled3D::ScalarVoxel<float>>("beam_angle_deg").get_data(), 12.5f);
}
