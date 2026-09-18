// CUDA device memory.
//
// The import *target*: a Vulkan or D3D12 allocation is imported and comes back as one of these, and
// an inference session binds it directly. `RFNN_WITH_CUDA` decides whether anything can produce
// one, but the type is always declared so a signature naming it still compiles.
#pragma once

#include <RadFiled3D/nn/memory/memory_ref.hpp>
#include <RadFiled3D/nn/memory/memory_ref.hpp>

namespace RadFiled3D::nn::memory::cuda {

/// Not `final`: an imported external allocation is a `cuda::MemoryRef` that additionally owns the
/// mapping, so a caller that needs the native pointer can cast to this type either way.
class MemoryRef : public memory::MemoryRef {
public:
    /// `device_ptr` is a `CUdeviceptr` — a 64-bit integer, so it crosses without cuda.h.
    ///
    /// `device` is the ordinal the allocation lives on. Pass it whenever it is known: it is what
    /// lets a session refuse memory from another GPU instead of handing the driver a pointer that is
    /// valid in a context it is not running in. -1 means "nobody said" — see `device_of`, which
    /// asks CUDA rather than guessing.
    MemoryRef(std::uint64_t device_ptr, std::uint64_t size_bytes, int device = -1) noexcept
        : device_ptr_(device_ptr), size_bytes_(size_bytes), device_(device) {}

    Domain get_domain() const noexcept override { return Domain::Cuda; }
    std::uint64_t get_size_bytes() const noexcept override { return size_bytes_; }

    /// Cast to `CUdeviceptr` on the caller's side.
    std::uint64_t get_address() const noexcept override { return device_ptr_; }
    int get_device_index() const noexcept override { return device_; }

    std::uint64_t get_device_ptr() const noexcept { return device_ptr_; }

private:
    std::uint64_t device_ptr_;
    std::uint64_t size_bytes_;
    int device_ = -1;
};

/// The CUDA device an existing device pointer lives on, or -1 if CUDA cannot say.
///
/// For memory this library did not allocate — a PyTorch tensor arriving through
/// `__cuda_array_interface__`, which carries an address and a length and no ordinal at all. Asking
/// the driver is the only way to know, and knowing is what turns "the device MUST match" from a
/// sentence in the documentation into a check.
int device_of(std::uint64_t device_ptr) noexcept;

/// Allocate device memory the reference OWNS — `cudaFree` on destruction.
///
/// What a stage's intermediates and weights live in when the session runs on a GPU, so a kernel can
/// read them directly and ONNX Runtime can bind them without a host round trip. Throws
/// `FeatureDisabled` without CUDA.
std::shared_ptr<memory::MemoryRef> allocate(std::uint64_t bytes, int device);

/// Copy within device memory. Both references must be `Domain::Cuda`.
///
/// The device-side equivalent of the `std::copy_n` that replicates a `Once` stage's single row
/// across a query batch; doing it host-side would mean two transfers per run of exactly the data
/// this design exists to keep on the GPU.
void copy_device_to_device(memory::MemoryRef& dst, std::uint64_t dst_offset, const memory::MemoryRef& src,
                           std::uint64_t src_offset, std::uint64_t bytes);

/// Import an external graphics allocation and map a buffer out of it.
///
/// This is the whole Vulkan/D3D12 → CUDA path, and it makes **no Vulkan or D3D call**:
/// `cudaImportExternalMemory` takes the file descriptor or NT handle the renderer already exported,
/// so importing needs the CUDA runtime and nothing else — no Vulkan loader, no SDK, nothing an
/// engine plugin has to ship on our behalf.
///
/// `device` is the CUDA ordinal the allocation lives on; pass what
/// `nn::cuda::get_device_for_uuid(buffer.device_uuid)` returned, so the memory and the session that
/// writes it are on the same GPU.
///
/// The returned reference OWNS the import: its destructor unmaps the buffer and destroys the
/// external-memory handle, so the `shared_ptr`'s lifetime is the mapping's lifetime.
///
/// Remember the handle asymmetry (`OpaqueFd` / `Win32Handle` in memory_ref.hpp): a successful
/// import consumes a POSIX fd and duplicates an NT handle.
///
/// Throws `FeatureDisabled` without CUDA, `InvalidArgument` on a bad device or an unusable handle.
std::shared_ptr<memory::MemoryRef> import_external_memory(const ExternalOrigin& buffer, int device);

/// Export memory THIS library allocated, so a graphics API can import it.
///
/// The other direction of `import_external_memory`, and the reason `allocate` uses the virtual
/// memory API: a `cudaMalloc` pointer has no shareable handle and could never be shown to a
/// renderer. This is what lets inference allocate its own output — because the caller bound none —
/// and still have Vulkan or D3D12 display it without a copy.
///
/// Works on any reference carrying a `VmmOrigin`, which includes a session's intermediate stage
/// buffers: they live as long as the model does, so exporting one is a legitimate way to watch what
/// a stage produced.
///
/// **The caller owns the returned handle.** On Windows it is an NT handle to close; on Linux a file
/// descriptor, and the importer consumes it. Throws `InvalidArgument` for memory that was imported
/// rather than allocated here, or has no exportable origin at all.
ExternalOrigin export_external_memory(const memory::MemoryRef& memory);

}  // namespace RadFiled3D::nn::memory::cuda
