#include <RadFiled3D/nn/memory/cuda.hpp>

#include <RadFiled3D/nn/backends/cuda.hpp>

#ifdef RFNN_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

namespace RadFiled3D::nn::memory::cuda {

#ifdef RFNN_WITH_CUDA
namespace {

/// The handle type an exported allocation can be imported through. A POSIX descriptor on Linux, an
/// NT handle on Windows — and the asymmetry matters downstream: CUDA duplicates an NT handle but
/// CONSUMES a descriptor, so only one of the two can be imported more than once.
#ifdef _WIN32
constexpr CUmemAllocationHandleType kShareableHandleType = CU_MEM_HANDLE_TYPE_WIN32;
#else
constexpr CUmemAllocationHandleType kShareableHandleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif

/// A driver-API error as text. The runtime API has `cudaGetErrorString`; the driver API needs two
/// calls and neither is guaranteed to answer, so an unknown code still yields its number.
std::string driver_error(CUresult rc) {
    const char* name = nullptr;
    const char* text = nullptr;
    cuGetErrorName(rc, &name);
    cuGetErrorString(rc, &text);
    std::string out = name != nullptr ? name : ("CUresult " + std::to_string(static_cast<int>(rc)));
    if (text != nullptr) out += std::string(" (") + text + ")";
    return out;
}

/// Make the device's PRIMARY context current on this thread.
///
/// Deliberately not `cuCtxCreate`: ONNX Runtime's CUDA provider allocates in the primary context, so
/// a context of our own would put its tensors and these allocations in different address spaces.
/// `backends/cuda.cpp` has the same helper for the same reason — it cannot be shared, because that
/// file is in `rfnn::inference` and this one is in `rfnn::core`, and core must not depend upward.
void use_primary_context(int device) {
    if (cudaSetDevice(device) != cudaSuccess)
        throw Exception::invalid_argument("no CUDA device " + std::to_string(device));
    if (cudaFree(nullptr) != cudaSuccess)
        throw Exception::invalid_argument("could not establish a CUDA context on device " +
                                          std::to_string(device));
}

/// A mapped external allocation. Holding the CUDA handles in the `MemoryRef` itself is what makes
/// the lifetime contract a destructor rather than a rule someone has to remember.
class ImportedMemoryRef final : public cuda::MemoryRef {
public:
    /// Keeps the ORIGIN, not just CUDA's own import object. Without it this reference is a dead end:
    /// there is no `cudaExportExternalMemory`, so a third API can only be given a view by importing
    /// the original exporter's handle again — and if we did not keep it, nobody can.
    ImportedMemoryRef(cudaExternalMemory_t external, void* mapped, std::uint64_t size_bytes, int device,
                      memory::ExternalOrigin origin)
        : cuda::MemoryRef(reinterpret_cast<std::uintptr_t>(mapped), size_bytes, device),
          external_(external), mapped_(mapped), device_(device), origin_(std::move(origin)) {}

    const memory::Origin& get_origin() const noexcept override { return origin_; }

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
    memory::Origin origin_;
};

/// A device allocation this reference owns, made through the VIRTUAL MEMORY API so it can be
/// exported.
///
/// `cudaMalloc` would be shorter, but its pointer has no shareable handle: nothing can ever be
/// given a view of it, so memory this library allocates could only ever be read back through a
/// copy. `cuMemCreate` with a requested handle type produces an allocation
/// `cuMemExportToShareableHandle` can hand to Vulkan or D3D12 — which is what makes "we allocated
/// the output because the caller bound none, and the renderer still displays it" possible.
///
/// The cost is that mapping is explicit: reserve address space, map the allocation into it, then
/// grant access. Teardown must undo exactly that, in reverse.
class OwnedMemoryRef final : public cuda::MemoryRef {
public:
    OwnedMemoryRef(CUdeviceptr device_ptr, std::uint64_t size_bytes, std::uint64_t reserved_bytes,
                   CUmemGenericAllocationHandle allocation, int device)
        : cuda::MemoryRef(static_cast<std::uint64_t>(device_ptr), size_bytes, device),
          pointer_(device_ptr), reserved_(reserved_bytes), allocation_(allocation), device_(device),
          origin_(memory::VmmOrigin{static_cast<std::uint64_t>(allocation), device}) {}

    ~OwnedMemoryRef() override {
        cudaSetDevice(device_);
        // Strict reverse order. Unmapping after releasing the handle, or freeing the address range
        // while it is still mapped, is undefined; errors are swallowed because a destructor has
        // nowhere to report them and the driver reclaims everything with the context anyway.
        if (pointer_ != 0 && reserved_ > 0) {
            cuMemUnmap(pointer_, reserved_);
            cuMemAddressFree(pointer_, reserved_);
        }
        if (allocation_ != 0) cuMemRelease(allocation_);
    }

    /// The allocation handle, so this memory can be exported to a graphics API.
    const memory::Origin& get_origin() const noexcept override { return origin_; }

private:
    CUdeviceptr pointer_ = 0;
    std::uint64_t reserved_ = 0;
    CUmemGenericAllocationHandle allocation_ = 0;
    int device_ = 0;
    memory::Origin origin_;
};

/// An empty allocation. A stage with no weights still HAS a buffer, so the reference must exist —
/// and there is nothing to reserve, map or export for zero bytes.
class EmptyMemoryRef final : public cuda::MemoryRef {
public:
    explicit EmptyMemoryRef(int device) : cuda::MemoryRef(0, 0, device) {}
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
    const int on = device < 0 ? 0 : device;
    if (cudaSetDevice(on) != cudaSuccess)
        throw Exception::invalid_argument("no CUDA device " + std::to_string(device));
    if (bytes == 0) return std::make_shared<EmptyMemoryRef>(on);
    use_primary_context(on);

    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = on;
    // Requesting an exportable handle at CREATION is the whole point — it cannot be added later,
    // and without it this allocation could never be shown to a renderer.
    prop.requestedHandleTypes = kShareableHandleType;

    std::size_t granularity = 0;
    if (cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS
        || granularity == 0)
        throw Exception::invalid_argument("cuMemGetAllocationGranularity failed");
    // The driver allocates and maps in whole granules (2 MiB on current hardware). The REQUESTED
    // size is what the reference reports, though: capacity is checked in elements (R-I1), and
    // reporting the padded size would silently inflate every `get_element_count`.
    const std::uint64_t reserved = ((bytes + granularity - 1) / granularity) * granularity;

    CUmemGenericAllocationHandle allocation = 0;
    if (const CUresult rc = cuMemCreate(&allocation, reserved, &prop, 0); rc != CUDA_SUCCESS)
        throw Exception::invalid_argument("cuMemCreate failed for " + std::to_string(reserved) +
                                          " bytes: " + driver_error(rc));

    CUdeviceptr pointer = 0;
    if (const CUresult rc = cuMemAddressReserve(&pointer, reserved, 0, 0, 0); rc != CUDA_SUCCESS) {
        cuMemRelease(allocation);
        throw Exception::invalid_argument(std::string("cuMemAddressReserve failed: ") + driver_error(rc));
    }
    if (const CUresult rc = cuMemMap(pointer, reserved, 0, allocation, 0); rc != CUDA_SUCCESS) {
        cuMemAddressFree(pointer, reserved);
        cuMemRelease(allocation);
        throw Exception::invalid_argument(std::string("cuMemMap failed: ") + driver_error(rc));
    }
    // Mapped is not readable: without this the first kernel or copy faults.
    CUmemAccessDesc access{};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (const CUresult rc = cuMemSetAccess(pointer, reserved, &access, 1); rc != CUDA_SUCCESS) {
        cuMemUnmap(pointer, reserved);
        cuMemAddressFree(pointer, reserved);
        cuMemRelease(allocation);
        throw Exception::invalid_argument(std::string("cuMemSetAccess failed: ") + driver_error(rc));
    }
    return std::make_shared<OwnedMemoryRef>(pointer, bytes, reserved, allocation, on);
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

ExternalOrigin export_external_memory([[maybe_unused]] const memory::MemoryRef& memory) {
#ifndef RFNN_WITH_CUDA
    throw Exception::feature_disabled("cuda", "exporting CUDA device memory");
#else
    const auto* vmm = std::get_if<memory::VmmOrigin>(&memory.get_origin());
    if (vmm == nullptr)
        throw Exception::invalid_argument(
            "this memory cannot be exported: it carries no allocation handle. Only memory allocated "
            "here is exportable — a reference IMPORTED from a graphics API must be shared by "
            "exporting another handle from the API that owns it, since CUDA cannot re-export.");

    use_primary_context(vmm->device);
    ExternalOrigin out;
    out.size_bytes = memory.get_size_bytes();
    out.offset_bytes = memory.get_offset_bytes();
    out.region_bytes = memory.get_size_bytes();

#ifdef _WIN32
    void* handle = nullptr;
    if (const CUresult rc = cuMemExportToShareableHandle(
            &handle, static_cast<CUmemGenericAllocationHandle>(vmm->allocation), kShareableHandleType, 0);
        rc != CUDA_SUCCESS)
        throw Exception::invalid_argument(std::string("cuMemExportToShareableHandle failed: ") +
                                          driver_error(rc));
    // An NT handle: the caller closes it, and CUDA duplicates rather than consuming on import, so
    // it may be handed to more than one importer.
    out.handle = Win32Handle{handle};
#else
    int fd = -1;
    if (const CUresult rc = cuMemExportToShareableHandle(
            &fd, static_cast<CUmemGenericAllocationHandle>(vmm->allocation), kShareableHandleType, 0);
        rc != CUDA_SUCCESS)
        throw Exception::invalid_argument(std::string("cuMemExportToShareableHandle failed: ") +
                                          driver_error(rc));
    // A descriptor the importer CONSUMES. One importer per export; a second view needs a second
    // call here, which is cheap.
    out.handle = OpaqueFd{fd};
#endif
    return out;
#endif
}

std::shared_ptr<memory::MemoryRef> import_external_memory([[maybe_unused]] const ExternalOrigin& buffer,
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
    // A POSIX descriptor is gone now — CUDA took ownership and closed it — while an NT handle was
    // duplicated and the exporter still holds its own. Recording which decides whether a third API
    // can be given a view of this allocation later, or has to be handed a freshly exported handle.
    memory::ExternalOrigin origin = buffer;
    origin.consumed = std::holds_alternative<memory::OpaqueFd>(buffer.handle);
    return std::make_shared<ImportedMemoryRef>(external, mapped, mapped_bytes, device,
                                               std::move(origin));
#endif
}

}  // namespace RadFiled3D::nn::memory::cuda
