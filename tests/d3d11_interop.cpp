// The D3D11 → CUDA/TensorRT path.
//
// D3D11 shares through CUDA's external-memory API just as Vulkan and D3D12 do — a shared handle
// imported once, giving a device pointer that stays valid. It differs in the import type, in having
// TWO handle kinds (the modern NT handle and the older global KMT handle), and in being dedicated
// always rather than only for a committed resource.
//
// A D3D11 shared handle only exists on Windows, so the import cannot run here. What is checked is
// everything around it, which is where silent breakage would live.
#include <RadFiled3D/nn/backends/compute_backend.hpp>
#include <RadFiled3D/nn/backends/cuda.hpp>
#include <RadFiled3D/nn/backends/dx11.hpp>
#include <RadFiled3D/nn/backends/dx12.hpp>
#include <RadFiled3D/nn/backends/tensorrt.hpp>
#include <RadFiled3D/nn/memory/dx11.hpp>
#include <RadFiled3D/nn/memory/dx12.hpp>

#include <gtest/gtest.h>

using namespace RadFiled3D::nn;

namespace {

memory::ExternalBuffer d3d11_buffer(memory::D3D11Handle::Kind kind) {
    memory::ExternalBuffer buffer;
    buffer.handle = memory::D3D11Handle{reinterpret_cast<void*>(std::uintptr_t{0xd3d11}), kind};
    buffer.size_bytes = 4096;
    buffer.region_bytes = 4096;
    return buffer;
}

}  // namespace

TEST(D3D11Interop, ReachesTheSameComputeBackendsAsD3D12) {
    memory::dx11::ExternalMemory interop;
    EXPECT_EQ(interop.get_domain(), memory::Domain::D3D11);

    // D3D11 is importable by the same compute backends, through the same external-memory API. If
    // this ever diverges from D3D12 it is a deliberate change, not an accident.
    const ComputeBackend& trt = get_compute_backend(Backend::TensorRt);
    EXPECT_TRUE(trt.can_import_from(memory::Domain::D3D11));
    EXPECT_EQ(trt.can_import_from(memory::Domain::D3D11), trt.can_import_from(memory::Domain::D3D12));
    // And it lands in CUDA memory, which is what a session binds.
    EXPECT_EQ(trt.get_memory_domain(), memory::Domain::Cuda);

    EXPECT_EQ(interop.supports(Backend::TensorRt), dx11::available() && tensorrt::available());
    EXPECT_EQ(interop.supports(Backend::Cuda), dx11::available() && cuda::available());
    // DirectML runs on D3D12; it cannot consume a D3D11 resource.
    EXPECT_FALSE(interop.supports(Backend::DirectMl));
    EXPECT_FALSE(interop.supports(Backend::Cpu));
}

TEST(D3D11Interop, ANullHandleIsRefusedForEitherKind) {
    if (!dx11::available() || !cuda::available()) GTEST_SKIP() << "D3D11 or CUDA compiled out";
    memory::dx11::ExternalMemory interop;
    for (const auto kind : {memory::D3D11Handle::Kind::NtHandle, memory::D3D11Handle::Kind::KmtHandle}) {
        memory::ExternalBuffer buffer = d3d11_buffer(kind);
        buffer.handle = memory::D3D11Handle{nullptr, kind};
        try {
            (void)interop.import_buffer(buffer, Backend::TensorRt);
            FAIL() << "a null D3D11 handle must not be imported";
        } catch (const Exception& err) {
            EXPECT_EQ(err.get_kind(), ErrorKind::InvalidArgument) << err.what();
            EXPECT_NE(std::string(err.what()).find("D3D11"), std::string::npos) << err.what();
        }
    }
}

/// Both handle kinds reach CUDA rather than one being rejected on the way — they are two ways of
/// sharing the same thing, not a supported one and an unsupported one.
TEST(D3D11Interop, BothHandleKindsReachTheImport) {
    if (!dx11::available() || !cuda::available()) GTEST_SKIP() << "D3D11 or CUDA compiled out";
    if (cuda::get_device_count() == 0) GTEST_SKIP() << "no CUDA device";
    memory::dx11::ExternalMemory interop;
    for (const auto kind : {memory::D3D11Handle::Kind::NtHandle, memory::D3D11Handle::Kind::KmtHandle}) {
        try {
            (void)interop.import_buffer(d3d11_buffer(kind), Backend::TensorRt);
            // Would only succeed against a real handle on Windows.
        } catch (const Exception& err) {
            EXPECT_EQ(err.get_kind(), ErrorKind::InvalidArgument) << err.what();
            EXPECT_NE(std::string(err.what()).find("cudaImportExternalMemory"), std::string::npos)
                << "expected the failure to come from CUDA, got: " << err.what();
        }
    }
}

/// An impossible pairing stays distinguishable from one that is merely not compiled in.
TEST(D3D11Interop, DirectMlCannotConsumeAD3D11Resource) {
    memory::dx11::ExternalMemory interop;
    try {
        (void)interop.import_buffer(d3d11_buffer(memory::D3D11Handle::Kind::NtHandle), Backend::DirectMl);
        FAIL() << "DirectML runs on D3D12 and cannot take a D3D11 resource";
    } catch (const Exception& err) {
        EXPECT_EQ(err.get_kind(), ErrorKind::UnsupportedInterop) << err.what();
    }
    // Where D3D12 under DirectML IS a meaningful pairing — the contrast is the point.
    memory::dx12::ExternalMemory d3d12;
    EXPECT_TRUE(get_compute_backend(Backend::DirectMl).can_import_from(memory::Domain::D3D12));
}
