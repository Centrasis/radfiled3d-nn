// Sharing GPU memory with a graphics API.
//
// The primary consumer of this library is a 3D engine (Unreal Engine 5): it owns the Vulkan or
// D3D12 device and has already allocated the buffer or volume texture its shader samples. Inference
// writes into that memory directly, so a predicted field reaches the renderer without a host
// round-trip (requirements.md §8).
//
// Interop is a PAIR, and the two halves are independent. Sharing memory takes a graphics API on
// one side and a compute backend on the other, chosen separately:
//
//   graphics ↓ / compute → | CUDA (NVIDIA)          | ROCm/HIP (AMD)         | DirectML           | CPU
//   Vulkan                 | external memory import | external memory import | ✗                  | ✗
//   D3D12                  | external memory import | external memory import | native — no import | ✗
//   D3D11                  | external memory import | external memory import | ✗                  | ✗
//
// Two consequences shape the API. Vulkan and D3D12 memory is shareable from an AMD card through HIP
// exactly as from an NVIDIA card through CUDA, so neither graphics backend implies `cuda`. And under
// the DirectML execution provider a D3D12 resource is already the provider's own memory, so the
// "import" is a no-op — the fast path on Windows needs no interop layer at all.
//
// Which pairing is in play is a RUNTIME question: the CMake options decide only which code is
// compiled, and a build may well carry several. `import_buffer` therefore takes the compute
// `Backend` and answers for that combination.
//
// Two rules bound this module:
//   * Memory sharing only. No device, queue, swapchain or pipeline is ever created here.
//   * Compiled out means an error, never a fallback. With the option off the API still exists and
//     throws `FeatureDisabled` (R-G3). An IMPOSSIBLE pairing is a different error,
//     `UnsupportedInterop`, so a configuration mistake is never confused with a design constraint.
#pragma once

#include <RadFiled3D/nn/core/session.hpp>
#include <RadFiled3D/nn/memory/memory_ref.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <variant>

namespace RadFiled3D::nn::memory {

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

/// An external allocation as the graphics API describes it, on the way in.
struct ExternalBuffer {
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
};

/// Imports and exports memory for one graphics API.
///
/// One interface covers both directions because a renderer needs both: importing the resource it
/// allocated as an inference target, and exporting a buffer this library owns for the renderer to
/// sample. Each graphics API derives it in its own sub-namespace (`memory::vk`, `memory::dx12`,
/// `memory::dx11`) — adding another means adding a file, never editing one.
class ExternalMemory {
public:
    virtual ~ExternalMemory() = default;

    /// The domain this backend imports FROM.
    virtual Domain get_domain() const noexcept = 0;

    /// Whether this graphics API can share memory with `compute` IN THIS BUILD: the pairing must be
    /// meaningful and both halves must be compiled in. Callers use it to choose a backend before
    /// allocating, rather than discovering the mismatch at bind time.
    virtual bool supports(Backend compute) const noexcept = 0;

    /// Import an engine allocation so inference can write into it.
    ///
    /// The returned reference is in the COMPUTE backend's domain once an import happened — a
    /// `memory::cuda::MemoryRef`, say — because that is what the session binds. Where no import is
    /// needed (a D3D12 resource under DirectML) it stays in the graphics domain and is bound as-is.
    ///
    /// The returned object OWNS the mapping: releasing the last `shared_ptr` releases the import.
    /// That is the whole lifetime contract — there is no separate close call to forget.
    virtual std::shared_ptr<MemoryRef> import_buffer(const ExternalBuffer& buffer, Backend compute) = 0;

    /// Export an allocation this library owns, for the graphics API to import.
    virtual ExternalBuffer export_buffer(std::uint64_t size_bytes, Backend compute) = 0;
};

}  // namespace RadFiled3D::nn::memory
