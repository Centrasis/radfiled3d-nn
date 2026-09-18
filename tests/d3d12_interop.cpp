// The D3D12 → CUDA/TensorRT path.
//
// A shared NT handle only exists on Windows, so the import itself cannot run here. What CAN be
// pinned down anywhere is everything around it — which pairings are possible, which are merely not
// compiled in, that a D3D12 reference is refused as a direct binding, and that the handle carries
// the resource/heap distinction CUDA insists on. Those are the parts that break silently.
#include <RadFiled3D/nn/backends/compute_backend.hpp>
#include <RadFiled3D/nn/backends/cuda.hpp>
#include <RadFiled3D/nn/backends/dx11.hpp>
#include <RadFiled3D/nn/backends/dx12.hpp>
#include <RadFiled3D/nn/backends/tensorrt.hpp>
#include <RadFiled3D/nn/memory/dx12.hpp>

#include <gtest/gtest.h>

using namespace RadFiled3D::nn;

namespace {

/// The graphics API this suite is about. Adoption is keyed on the domain, so it is named
/// once here rather than at every call.
constexpr auto kDomain = memory::Domain::D3D12;


/// A handle value that is not null but names nothing — enough to get past the null check and reach
/// the import, which is as far as any non-Windows host can go.
void* fake_handle() { return reinterpret_cast<void*>(std::uintptr_t{0xd3d12}); }

memory::ExternalOrigin d3d12_buffer(memory::D3D12Handle::Kind kind) {
    memory::ExternalOrigin buffer;
    buffer.handle = memory::D3D12Handle{fake_handle(), kind};
    buffer.size_bytes = 4096;
    buffer.region_bytes = 4096;
    return buffer;
}

}  // namespace

/// TensorRT shares CUDA's memory exactly, so whatever CUDA can import from, TensorRT can too. If
/// these ever disagree, one of them grew its own opinion and the delegation broke.
TEST(D3D12Interop, TensorRtImportsThroughCudasMemory) {
    const ComputeBackend& cuda_backend = get_compute_backend(Backend::Cuda);
    const ComputeBackend& trt = get_compute_backend(Backend::TensorRt);

    EXPECT_EQ(trt.get_memory_domain(), cuda_backend.get_memory_domain());
    EXPECT_EQ(trt.get_ort_allocator_name(), cuda_backend.get_ort_allocator_name());
    for (const auto domain : {memory::Domain::D3D12, memory::Domain::D3D11, memory::Domain::Vulkan,
                              memory::Domain::Host, memory::Domain::Cuda})
        EXPECT_EQ(trt.can_import_from(domain), cuda_backend.can_import_from(domain))
            << "disagreement on " << memory::to_string(domain);

    EXPECT_TRUE(trt.can_import_from(memory::Domain::D3D12));
    // The reference a D3D12 import yields is CUDA memory — that is what a session binds.
    EXPECT_EQ(trt.get_memory_domain(), memory::Domain::Cuda);
}

TEST(D3D12Interop, ThePairingIsPossibleAndTheOptionIsWhatGatesIt) {
    EXPECT_EQ(kDomain, memory::Domain::D3D12);

    // Possible in principle; whether it works here is a build question, and the two are different
    // questions on purpose.
    EXPECT_TRUE(get_compute_backend(Backend::TensorRt).can_import_from(memory::Domain::D3D12));
    EXPECT_EQ((get_compute_backend(Backend::TensorRt).can_import_from(kDomain) && is_available(Backend::TensorRt)), dx12::available() && tensorrt::available());
    EXPECT_EQ((get_compute_backend(Backend::Cuda).can_import_from(kDomain) && is_available(Backend::Cuda)), dx12::available() && cuda::available());

    // The CPU provider is not a configuration mistake — no build option would ever make it work.
    EXPECT_FALSE((get_compute_backend(Backend::Cpu).can_import_from(kDomain) && is_available(Backend::Cpu)));
    try {
        (void)get_compute_backend(Backend::Cpu).adopt(std::make_shared<memory::ExportedMemoryRef>(kDomain, d3d12_buffer(memory::D3D12Handle::Kind::Resource)), -1);
        FAIL() << "a D3D12 resource cannot reach the CPU provider";
    } catch (const Exception& err) {
        EXPECT_EQ(err.get_kind(), ErrorKind::UnsupportedInterop) << err.what();
    }
}

/// A null handle is caught before CUDA sees it, whichever kind it claims to be.
TEST(D3D12Interop, ANullHandleIsRefused) {
    if (!dx12::available() || !cuda::available()) GTEST_SKIP() << "D3D12 or CUDA compiled out";
    for (const auto kind : {memory::D3D12Handle::Kind::Resource, memory::D3D12Handle::Kind::Heap}) {
        memory::ExternalOrigin buffer = d3d12_buffer(kind);
        buffer.handle = memory::D3D12Handle{nullptr, kind};
        try {
            (void)get_compute_backend(Backend::TensorRt).adopt(std::make_shared<memory::ExportedMemoryRef>(kDomain, buffer), -1);
            FAIL() << "a null D3D12 handle must not be imported";
        } catch (const Exception& err) {
            EXPECT_EQ(err.get_kind(), ErrorKind::InvalidArgument) << err.what();
            EXPECT_NE(std::string(err.what()).find("D3D12"), std::string::npos) << err.what();
        }
    }
}

/// Both kinds reach the import and fail there, not earlier — a heap is as valid a thing to hand
/// over as a resource, and the two take different CUDA paths rather than one being rejected.
TEST(D3D12Interop, AResourceAndAHeapAreBothAccepted) {
    if (!dx12::available() || !cuda::available()) GTEST_SKIP() << "D3D12 or CUDA compiled out";
    if (cuda::get_device_count() == 0) GTEST_SKIP() << "no CUDA device";
    for (const auto kind : {memory::D3D12Handle::Kind::Resource, memory::D3D12Handle::Kind::Heap}) {
        try {
            (void)get_compute_backend(Backend::TensorRt).adopt(std::make_shared<memory::ExportedMemoryRef>(kDomain, d3d12_buffer(kind)), -1);
            // Would only succeed against a real handle on Windows.
        } catch (const Exception& err) {
            // The failure must come from CUDA rejecting the handle, never from this library
            // refusing the KIND — the whole point is that both are importable.
            EXPECT_EQ(err.get_kind(), ErrorKind::InvalidArgument) << err.what();
            EXPECT_NE(std::string(err.what()).find("cudaImportExternalMemory"), std::string::npos)
                << "expected the failure to come from CUDA, got: " << err.what();
        }
    }
}
