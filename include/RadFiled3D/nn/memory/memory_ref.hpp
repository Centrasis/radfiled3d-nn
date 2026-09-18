// A reference to a block of memory, wherever it lives.
//
// `MemoryRef` is the only currency in which `RadFiled3D::nn` talks about buffers. A caller binds one
// to an inference input, a field layer keeps one as its GPU mirror, and an imported graphics
// allocation hands one out — and none of those places learns whether the memory behind it is a
// CUDA device pointer, a `VkBuffer`, an `ID3D12Resource` or a plain host pointer.
//
// The base is abstract and each backend specialises it in its own sub-namespace
// (`memory::cuda`, `memory::vk`, `memory::dx12`, …), mirroring how RadFiled3D declares an interface
// in `Storage` and derives the concrete type in `Storage::V1`. The specialisation is not
// decoration: the backends genuinely carry different handles. CUDA needs one device pointer, Vulkan
// needs a `VkBuffer` *and* the `VkDeviceMemory` it is bound to, D3D12 needs an `ID3D12Resource`.
// One flat struct could not hold them without pretending they are the same shape.
//
// No header here includes a GPU SDK: every native handle crosses as a 64-bit value, which is what
// `CUdeviceptr`, a dispatchable Vulkan handle and a COM pointer all are. A consumer that has the SDK
// casts on its own side. That is what lets an engine plugin include these headers without the CUDA
// toolkit, and what lets a build with a backend compiled out still name its types.
//
// OWNERSHIP IS THE SUBCLASS'S BUSINESS, not the interface's. `host::MemoryRef` wraps a pointer it
// does not own; `cuda::OwnedMemoryRef` frees what it allocated; an imported reference releases the
// mapping and the external-memory object it holds. All three are the same type to a caller, which is
// the point — the alternative is an owning/non-owning split in the public type, and then every
// signature has to say which it wants. Releasing the last `shared_ptr` does the right thing in each
// case, and for a reference that owns nothing the allocation must still outlive it.
#pragma once

#include <RadFiled3D/nn/exception.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <variant>

namespace RadFiled3D::nn::memory {

/// Which allocator owns the memory, and therefore who may interpret its handles. The set is closed:
/// a domain the runtime cannot name is memory it cannot safely touch.
enum class Domain : std::uint8_t {
    /// Ordinary process memory.
    Host,
    /// CUDA device memory (NVIDIA).
    Cuda,
    /// HIP/ROCm device memory (AMD).
    Hip,
    /// A Vulkan buffer.
    Vulkan,
    /// A Direct3D 11 resource, shared through CUDA's external-memory path as an NT or KMT handle.
    D3D11,
    /// A Direct3D 12 resource or heap. Shared through the modern external-memory path.
    D3D12,
};

std::string_view to_string(Domain domain) noexcept;

/// `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`, from `vkGetMemoryFdKHR`.
///
/// **The importer CONSUMES it.** CUDA takes ownership of the descriptor on a successful import and
/// closes it itself; closing it again is a double free. This is the opposite of `Win32Handle` below,
/// and the asymmetry is the platform's, not ours.
struct OpaqueFd {
    int fd = -1;
};
/// `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT`, or a D3D11/D3D12 shared NT handle.
///
/// **The caller keeps it.** CUDA duplicates an NT handle rather than taking it, so the exporter must
/// close its own handle once the import is done with it — and must keep it open until then.
struct Win32Handle {
    void* handle = nullptr;
};

/// A Direct3D 12 shared NT handle from `ID3D12Device::CreateSharedHandle`, and WHAT it names.
///
/// A heap and a committed resource are NOT interchangeable: they take different CUDA import types,
/// and CUDA **requires** the dedicated flag for a resource while a heap must not carry it. Getting
/// that pair wrong fails the import with a message that says nothing useful, so the kind travels
/// with the handle instead of being guessed from anything.
///
/// Kept separate from `Win32Handle` for the same reason: Vulkan's `OPAQUE_WIN32` export and a D3D12
/// shared handle are both NT handles and both `void*`, but they import as different types. One
/// alternative for both would make that distinction unrepresentable.
struct D3D12Handle {
    void* handle = nullptr;
    enum class Kind : std::uint8_t {
        /// An `ID3D12Resource` — a committed resource, imported as a dedicated allocation.
        Resource,
        /// An `ID3D12Heap` the renderer suballocates from; `offset_bytes` selects the region.
        Heap,
    };
    Kind kind = Kind::Resource;
};

/// A Direct3D 11 shared handle, and WHICH KIND of shared handle it is.
///
/// D3D11 has two sharing mechanisms and they are not interchangeable: the modern NT handle from
/// `IDXGIResource1::CreateSharedHandle` (needs `D3D11_RESOURCE_MISC_SHARED_NTHANDLE`) and the older
/// global KMT handle from `IDXGIResource::GetSharedHandle` (needs `D3D11_RESOURCE_MISC_SHARED`).
/// CUDA imports them as different types, so the kind travels with the handle.
///
/// Both import as DEDICATED — CUDA requires it for every D3D11 resource, unlike D3D12 where only a
/// committed resource is dedicated.
struct D3D11Handle {
    void* handle = nullptr;
    enum class Kind : std::uint8_t {
        /// `IDXGIResource1::CreateSharedHandle`. An NT handle: CUDA duplicates it and the caller
        /// keeps ownership.
        NtHandle,
        /// `IDXGIResource::GetSharedHandle`. A global KMT handle, not an NT handle — there is no
        /// ownership to transfer and nothing for the caller to close.
        KmtHandle,
    };
    Kind kind = Kind::NtHandle;
};

/// An `ID3D12Resource*` on the SAME device the session runs on — the object itself, not a handle to
/// it.
///
/// This is what the DirectML path takes, and it is a different thing from `D3D12Handle` rather than
/// a variation on it. DirectML executes on D3D12, so a resource the caller already owns needs no
/// sharing, no export and no import: there is nothing to cross. Giving it a shared handle instead
/// would force this library to call `ID3D12Device::OpenSharedHandle` — a Direct3D call, which is
/// precisely what it does not make.
///
/// The caller keeps the COM reference; nothing here AddRefs it.
struct D3D12NativeResource {
    void* resource = nullptr;
};

/// A platform handle to memory allocated by a graphics API. The alternatives are per-platform and
/// neither leaks into the common API.
using ExternalHandle = std::variant<OpaqueFd, Win32Handle, D3D12Handle, D3D11Handle, D3D12NativeResource>;


/// Imports and exports memory for one graphics API.
///
/// One interface covers both directions because a renderer needs both: importing the resource it
/// allocated as an inference target, and exporting a buffer this library owns for the renderer to
/// sample. Each graphics API derives it in its own sub-namespace (`memory::vk`, `memory::dx12`,
/// `memory::dx11`) — adding another means adding a file, never editing one.

/// WHERE A REFERENCE'S MEMORY CAME FROM, when it can be shared with a third API.
///
/// This exists for one case, and it is the case zero-copy interop is for: memory a renderer
/// allocated, imported into a compute backend, and then wanted as a view in a SECOND API — D3D12
/// allocates, CUDA computes, Vulkan renders. That chain is not a chain. There is no
/// `cudaExportExternalMemory` and a `cudaMalloc` pointer has no shareable handle, so the third view
/// cannot come from the CUDA reference; it must be imported from the ORIGINAL exporter's handle.
/// A reference therefore has to remember where it came from, or the third view is not expressible
/// at all.
///
/// The two origins are NOT interchangeable and must not be flattened into one "shareable handle":
/// an imported reference can only offer the foreign handle it was built from, while one this
/// library allocated exportably can offer its own.
struct NoOrigin {};

/// Another API allocated this and exported a handle; we imported it.
///
/// **Whether the handle can be imported AGAIN is the platform's rule, not ours.** CUDA duplicates an
/// NT handle (`Win32Handle`, and so every D3D11/D3D12 share), so a second and third importer are
/// fine as long as the exporter holds its handle open. CUDA CONSUMES a POSIX file descriptor
/// (`OpaqueFd`, the Vulkan path on Linux) and closes it, so a further view needs the application to
/// export another one. `consumed` records which happened, so a later `adopt` can say that rather
/// than failing on a recycled descriptor.
struct ExternalOrigin {
    ExternalHandle handle;
    /// Size of the WHOLE external allocation, not of the region — this is what the importing API is
    /// told, and it must match what the exporter allocated.
    std::uint64_t size_bytes = 0;
    /// Offset of the region to use within that allocation, and its length. A renderer suballocates,
    /// so the buffer is rarely the whole heap. `region_bytes == 0` means "to the end".
    std::uint64_t offset_bytes = 0;
    std::uint64_t region_bytes = 0;
    /// The device the allocation lives on: `VkPhysicalDeviceIDProperties::deviceUUID`, which is the
    /// same 16 bytes CUDA reports as `cudaDeviceProp::uuid`. All-zero means "the current device",
    /// which keeps a single-GPU caller free of ceremony. Binding memory from one device into a
    /// session on another is refused rather than silently copied (R-I1).
    std::array<std::uint8_t, 16> device_uuid{};
    /// True when the allocation was made with `VkMemoryDedicatedAllocateInfo`. The import must be
    /// told, or CUDA rejects or mis-maps it.
    ///
    /// Applies to a Vulkan import. A `D3D12Handle` decides it from its own `kind` — CUDA leaves no
    /// choice there — so this field is ignored for one.
    bool dedicated = false;

    /// The length actually mapped: `region_bytes`, or the rest of the allocation.
    std::uint64_t mapped_bytes() const noexcept {
        return region_bytes != 0 ? region_bytes : (size_bytes > offset_bytes ? size_bytes - offset_bytes : 0);
    }
    /// Set once an importer has taken the handle: see the platform note above. Not part of the
    /// description, but of what has since happened to it.
    bool consumed = false;
};

/// This library allocated it with an exportable handle (the CUDA virtual-memory API), so it can be
/// handed OUT to a graphics API rather than only received from one.
struct VmmOrigin {
    /// A `CUmemGenericAllocationHandle`, as a plain 64-bit value like every other native handle here.
    std::uint64_t allocation = 0;
    int device = -1;
};

using Origin = std::variant<NoOrigin, ExternalOrigin, VmmOrigin>;

/// The type-erased reference `RadFiled3D::nn` passes around.
class MemoryRef {
public:
    virtual ~MemoryRef() = default;

    MemoryRef(const MemoryRef&) = delete;
    MemoryRef& operator=(const MemoryRef&) = delete;

    virtual Domain get_domain() const noexcept = 0;
    virtual std::uint64_t get_size_bytes() const noexcept = 0;

    /// The address or handle this reference denotes, as a plain 64-bit value.
    ///
    /// This is the type-erased view the typed accessors (`get_pointer`, `get_device_ptr`,
    /// `get_buffer`, `get_resource`) each specialise. A runtime needs an address to hand to its API
    /// and must not have to `dynamic_cast` through every backend to find one — that would put a list
    /// of backends in the runtime, which is exactly what `MemoryRef` exists to avoid. It stays
    /// opaque: only the owning backend may interpret it.
    virtual std::uint64_t get_address() const noexcept = 0;

    /// Where this memory came from, for a backend that wants its own view of it. `NoOrigin` is the
    /// ordinary answer — host memory, and any device allocation made without an exportable handle —
    /// and it means no other API can be given a view of this without a copy.
    virtual const Origin& get_origin() const noexcept {
        static const Origin kNone{};
        return kNone;
    }

    /// Record that an importer has taken this reference's handle.
    ///
    /// Only a POSIX descriptor is consumed, and only the reference that HOLDS the handle can know
    /// it happened — `adopt` calls this after a successful import so a second attempt reports the
    /// platform rule rather than failing inside the driver on a descriptor that is already closed.
    /// A no-op for everything else, which is most references.
    virtual void mark_origin_consumed() noexcept {}
    /// Offset within the allocation the handles name; graphics APIs suballocate. Zero unless the
    /// backend says otherwise.
    virtual std::uint64_t get_offset_bytes() const noexcept { return 0; }

    /// Which device OF ITS OWN DOMAIN this memory lives on, or -1 when there is no such thing.
    ///
    /// -1 means one of two honest answers: the domain has no ordinal (host memory binds to a session
    /// on any device; a graphics allocation is identified by UUID, not by index), or the reference
    /// was built from a bare pointer and nobody said. It is NEVER a synonym for device 0 — a session
    /// checking that bound memory is on its own device must not accept "unknown" as "mine" and must
    /// not reject it either, because a host buffer would then be unbindable.
    virtual int get_device_index() const noexcept { return -1; }

    bool is_host() const noexcept { return get_domain() == Domain::Host; }

    /// How many elements of `element_bytes` fit. Capacity is checked in elements, never in bytes
    /// (R-I1), and this is what that check divides by.
    std::uint64_t get_element_count(std::uint64_t element_bytes) const;

    /// Throws `InvalidArgument` unless this reference is in `expected` — the check every backend
    /// runs before it casts a handle it is only allowed to interpret for its own domain.
    void require_domain(Domain expected) const;

protected:
    MemoryRef() = default;
};

namespace host {

/// Ordinary process memory. What a caller binds when it hands over a `std::vector` or a NumPy
/// array, and the only `MemoryRef` that exists in a build with no GPU backend at all.
class MemoryRef final : public memory::MemoryRef {
public:
    MemoryRef(void* data, std::uint64_t size_bytes) noexcept : data_(data), size_bytes_(size_bytes) {}

    /// Reference a contiguous range. `T` is erased; only the byte extent survives.
    template <typename T>
    static std::shared_ptr<MemoryRef> of(std::span<T> values) {
        return std::make_shared<MemoryRef>(const_cast<void*>(static_cast<const void*>(values.data())),
                                           values.size() * sizeof(T));
    }

    Domain get_domain() const noexcept override { return Domain::Host; }
    std::uint64_t get_size_bytes() const noexcept override { return size_bytes_; }
    std::uint64_t get_address() const noexcept override {
        return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data_));
    }

    void* get_pointer() const noexcept { return data_; }

private:
    void* data_;
    std::uint64_t size_bytes_;
};

/// Allocate ordinary process memory the reference OWNS.
///
/// `MemoryRef::of` BORROWS a caller's buffer; this one keeps its own, which is what an intermediate
/// between stages needs — nobody else owns it and it must outlive every binding into it.
std::shared_ptr<memory::MemoryRef> allocate(std::uint64_t bytes);

}  // namespace host

/// SHARING MEMORY WITH A GRAPHICS API — the pairing matrix.
///
/// Interop is a PAIR, and the two halves are independent. Sharing takes a graphics API on one side
/// and a compute backend on the other, chosen separately:
///
///   graphics | CUDA (NVIDIA)          | ROCm/HIP (AMD)         | DirectML           | CPU
///   Vulkan   | external memory import | external memory import | x                  | x
///   D3D12    | external memory import | external memory import | native - no import | x
///   D3D11    | external memory import | external memory import | x                  | x
///
/// Two consequences shape the API. Vulkan and D3D12 memory is shareable from an AMD card through
/// HIP exactly as from an NVIDIA card through CUDA, so neither graphics backend implies `cuda`. And
/// under the DirectML execution provider a D3D12 resource is already the provider's own memory, so
/// the "import" is a no-op — the fast path on Windows needs no interop layer at all.
///
/// Which pairing is in play is a RUNTIME question: the CMake options decide only which code is
/// compiled, and a build may well carry several. `ComputeBackend::adopt` therefore answers for the
/// combination in front of it, and tells an IMPOSSIBLE pairing (`UnsupportedInterop`) from one that
/// is merely not compiled in (`FeatureDisabled`) — a configuration mistake must never be confused
/// with a design constraint.
///
/// Memory sharing only: no device, queue, swapchain or pipeline is ever created here.
/// An allocation another API owns and has EXPORTED a handle for, before anything imported it.
///
/// This is what a caller hands in, and it is an ordinary `MemoryRef` so that every signature in the
/// library speaks one type. Binding it does the right thing: `ComputeBackend::adopt` sees a domain
/// that is not the backend's own, finds the `ExternalOrigin`, and imports — so a caller never calls
/// an import function, never holds a descriptor type, and never branches on what it has.
///
/// It is NOT addressable. `get_address()` returns the native handle, not a pointer, because there
/// is nothing to dereference until an import happens; that is also why nothing in the library reads
/// through a reference whose domain it does not own (`require_domain` is the guard).
///
/// **The renderer still exports.** This library calls no Vulkan, D3D11 or D3D12 entry point — it
/// links neither loader nor SDK, which is what lets an engine plugin include these headers without
/// them — so obtaining the handle is the owning API's job. What this type removes is everything
/// after that.
class ExportedMemoryRef final : public MemoryRef {
public:
    /// `domain` is the API that owns the allocation: `Domain::Vulkan`, `Domain::D3D12`,
    /// `Domain::D3D11`.
    ExportedMemoryRef(Domain domain, ExternalOrigin origin) noexcept
        : domain_(domain), origin_(std::move(origin)) {}

    Domain get_domain() const noexcept override { return domain_; }
    std::uint64_t get_size_bytes() const noexcept override { return external().mapped_bytes(); }
    std::uint64_t get_offset_bytes() const noexcept override { return external().offset_bytes; }
    /// The native handle, not an address — see the note above.
    std::uint64_t get_address() const noexcept override;
    /// A graphics allocation is identified by UUID, not by a compute ordinal, so there is no index
    /// to report until it has been imported onto a device.
    int get_device_index() const noexcept override { return -1; }

    const Origin& get_origin() const noexcept override { return origin_; }
    void mark_origin_consumed() noexcept override {
        std::get<ExternalOrigin>(origin_).consumed = true;
    }

    /// The description this was built from.
    const ExternalOrigin& external() const noexcept { return std::get<ExternalOrigin>(origin_); }

private:
    Domain domain_;
    Origin origin_;
};

}  // namespace RadFiled3D::nn::memory
