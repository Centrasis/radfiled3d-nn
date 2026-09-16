#include <RadFiled3D/nn/memory/cuda.hpp>

#include <RadFiled3D/nn/backends/cuda.hpp>

#ifdef RFNN_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace RadFiled3D::nn::memory::cuda {

#ifdef RFNN_WITH_CUDA
namespace {

/// A mapped external allocation. Holding the CUDA handles in the `MemoryRef` itself is what makes
/// the lifetime contract a destructor rather than a rule someone has to remember.
class ImportedMemoryRef final : public cuda::MemoryRef {
public:
    ImportedMemoryRef(cudaExternalMemory_t external, void* mapped, std::uint64_t size_bytes, int device)
        : cuda::MemoryRef(reinterpret_cast<std::uintptr_t>(mapped), size_bytes, device),
          external_(external), mapped_(mapped), device_(device) {}

    ~ImportedMemoryRef() override {
        // Order matters: the mapped pointer must go before the handle that produced it. Errors are
        // swallowed because a destructor has nowhere to report them and the process is not harmed —
        // the driver reclaims everything when the context goes.
        cudaSetDevice(device_);
        if (mapped_ != nullptr) cudaFree(mapped_);
        if (external_ != nullptr) cudaDestroyExternalMemory(external_);
    }

private:
    cudaExternalMemory_t external_ = nullptr;
    void* mapped_ = nullptr;
    int device_ = 0;
};

/// A plain device allocation this reference owns.
class OwnedMemoryRef final : public cuda::MemoryRef {
public:
    OwnedMemoryRef(void* device_ptr, std::uint64_t size_bytes, int device)
        : cuda::MemoryRef(reinterpret_cast<std::uintptr_t>(device_ptr), size_bytes, device),
          pointer_(device_ptr), device_(device) {}

    ~OwnedMemoryRef() override {
        cudaSetDevice(device_);
        if (pointer_ != nullptr) cudaFree(pointer_);
    }

private:
    void* pointer_ = nullptr;
    int device_ = 0;
};

}  // namespace
#endif

int device_of([[maybe_unused]] std::uint64_t device_ptr) noexcept {
#ifndef RFNN_WITH_CUDA
    return -1;
#else
    if (device_ptr == 0) return -1;
    cudaPointerAttributes attributes{};
    if (cudaGetLastError() != cudaSuccess) { /* clear a sticky error so ours is meaningful */ }
    if (cudaPointerGetAttributes(&attributes,
                                 reinterpret_cast<const void*>(static_cast<std::uintptr_t>(device_ptr))) !=
        cudaSuccess) {
        // A pointer CUDA does not recognise is not an error here: the caller may simply have handed
        // over host memory, and the domain check catches that with a better message.
        cudaGetLastError();
        return -1;
    }
    return attributes.type == cudaMemoryTypeUnregistered ? -1 : attributes.device;
#endif
}

std::shared_ptr<memory::MemoryRef> allocate([[maybe_unused]] std::uint64_t bytes,
                                            [[maybe_unused]] int device) {
#ifndef RFNN_WITH_CUDA
    throw Exception::feature_disabled("cuda", "allocating CUDA device memory");
#else
    if (cudaSetDevice(device < 0 ? 0 : device) != cudaSuccess)
        throw Exception::invalid_argument("no CUDA device " + std::to_string(device));
    void* pointer = nullptr;
    // A zero-length allocation is legal and useful: a stage with no weights still HAS the buffer, so
    // the reference must exist. `cudaMalloc(0)` yields a null pointer, which nothing dereferences
    // because nothing reads zero bytes.
    if (bytes > 0 && cudaMalloc(&pointer, static_cast<std::size_t>(bytes)) != cudaSuccess)
        throw Exception::invalid_argument("cudaMalloc failed for " + std::to_string(bytes) + " bytes");
    return std::make_shared<OwnedMemoryRef>(pointer, bytes, device < 0 ? 0 : device);
#endif
}

void copy_device_to_device([[maybe_unused]] memory::MemoryRef& dst, [[maybe_unused]] std::uint64_t dst_offset,
                           [[maybe_unused]] const memory::MemoryRef& src,
                           [[maybe_unused]] std::uint64_t src_offset, [[maybe_unused]] std::uint64_t bytes) {
#ifndef RFNN_WITH_CUDA
    throw Exception::feature_disabled("cuda", "copying within CUDA device memory");
#else
    if (bytes == 0) return;
    dst.require_domain(Domain::Cuda);
    src.require_domain(Domain::Cuda);
    if (dst_offset + bytes > dst.get_size_bytes() || src_offset + bytes > src.get_size_bytes())
        throw Exception::invalid_argument("device copy would run past the end of an allocation");
    const auto address = [](const memory::MemoryRef& ref, std::uint64_t offset) {
        return reinterpret_cast<void*>(static_cast<std::uintptr_t>(ref.get_address() + ref.get_offset_bytes() +
                                                                   offset));
    };
    // Issue the copy in the context the memory belongs to. A session on device 1 that left device 0
    // current would copy in the wrong context — which on a single-GPU machine is indistinguishable
    // from working.
    if (dst.get_device_index() >= 0) cudaSetDevice(dst.get_device_index());
    if (cudaMemcpy(address(dst, dst_offset), address(src, src_offset), static_cast<std::size_t>(bytes),
                   cudaMemcpyDeviceToDevice) != cudaSuccess)
        throw Exception::invalid_argument("cudaMemcpy device-to-device failed");
#endif
}

std::shared_ptr<memory::MemoryRef> import_external_memory([[maybe_unused]] const ExternalBuffer& buffer,
                                                          [[maybe_unused]] int device) {
#ifndef RFNN_WITH_CUDA
    throw Exception::feature_disabled("cuda", "importing an external graphics allocation into CUDA");
#else
    if (device < 0)
        throw Exception::invalid_argument(
            "no CUDA device matches the allocation's Vulkan/D3D device UUID; binding memory from one "
            "device into a session on another is refused rather than silently copied");
    nn::cuda::set_device(device);

    const std::uint64_t mapped_bytes = buffer.mapped_bytes();
    if (mapped_bytes == 0)
        throw Exception::invalid_argument("external allocation maps zero bytes at offset " +
                                          std::to_string(buffer.offset_bytes) + " of " +
                                          std::to_string(buffer.size_bytes));

    cudaExternalMemoryHandleDesc desc{};
    desc.size = buffer.size_bytes;
    // A Vulkan allocation made with VkMemoryDedicatedAllocateInfo must be imported as dedicated, or
    // CUDA refuses it — the flag is not cosmetic. A D3D12 handle overrides this below, because there
    // CUDA dictates the flag rather than the caller.
    desc.flags = buffer.dedicated ? cudaExternalMemoryDedicated : 0u;

    if (const auto* fd = std::get_if<OpaqueFd>(&buffer.handle)) {
        if (fd->fd < 0) throw Exception::invalid_argument("external memory: invalid file descriptor");
        desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        desc.handle.fd = fd->fd;
    } else if (const auto* win32 = std::get_if<Win32Handle>(&buffer.handle)) {
        if (win32->handle == nullptr) throw Exception::invalid_argument("external memory: null Win32 handle");
        desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        desc.handle.win32.handle = win32->handle;
    } else if (const auto* d3d12 = std::get_if<D3D12Handle>(&buffer.handle)) {
        if (d3d12->handle == nullptr) throw Exception::invalid_argument("external memory: null D3D12 handle");
        const bool is_resource = d3d12->kind == D3D12Handle::Kind::Resource;
        desc.type = is_resource ? cudaExternalMemoryHandleTypeD3D12Resource
                                : cudaExternalMemoryHandleTypeD3D12Heap;
        desc.handle.win32.handle = d3d12->handle;
        // Not the caller's choice: CUDA REQUIRES the dedicated flag for a committed resource and
        // must not see it for a heap. Deriving it from the kind is the only way to be right, and
        // the alternative is an import that fails with nothing to go on.
        desc.flags = is_resource ? cudaExternalMemoryDedicated : 0u;
    } else if (const auto* d3d11 = std::get_if<D3D11Handle>(&buffer.handle)) {
        if (d3d11->handle == nullptr) throw Exception::invalid_argument("external memory: null D3D11 handle");
        desc.type = d3d11->kind == D3D11Handle::Kind::NtHandle
                        ? cudaExternalMemoryHandleTypeD3D11Resource
                        : cudaExternalMemoryHandleTypeD3D11ResourceKmt;
        desc.handle.win32.handle = d3d11->handle;
        // Every D3D11 resource imports as dedicated — CUDA requires it for both handle kinds, where
        // D3D12 requires it only for a committed resource.
        desc.flags = cudaExternalMemoryDedicated;
    } else if (std::holds_alternative<D3D12NativeResource>(buffer.handle)) {
        // That alternative exists for DirectML, which shares CUDA's problem in reverse: it needs no
        // import at all. CUDA is a different device API and can only take something exported.
        throw Exception::invalid_argument(
            "CUDA needs a SHARED handle (memory::D3D12Handle from ID3D12Device::CreateSharedHandle); "
            "an ID3D12Resource pointer is only usable by a backend running on that same D3D12 device");
    } else {
        throw Exception::invalid_argument("external memory: no handle");
    }

    cudaExternalMemory_t external = nullptr;
    if (const cudaError_t rc = cudaImportExternalMemory(&external, &desc); rc != cudaSuccess)
        throw Exception::invalid_argument(std::string("cudaImportExternalMemory failed: ") +
                                          cudaGetErrorString(rc));

    cudaExternalMemoryBufferDesc region{};
    region.offset = buffer.offset_bytes;
    region.size = mapped_bytes;
    region.flags = 0;

    void* mapped = nullptr;
    if (const cudaError_t rc = cudaExternalMemoryGetMappedBuffer(&mapped, external, &region);
        rc != cudaSuccess) {
        cudaDestroyExternalMemory(external);
        throw Exception::invalid_argument(std::string("cudaExternalMemoryGetMappedBuffer failed: ") +
                                          cudaGetErrorString(rc));
    }
    return std::make_shared<ImportedMemoryRef>(external, mapped, mapped_bytes, device);
#endif
}

}  // namespace RadFiled3D::nn::memory::cuda
