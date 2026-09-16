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
// It is a REFERENCE: it allocates nothing, frees nothing, and extends no lifetime. Whoever owns the
// allocation must outlive every `MemoryRef` handed out for it.
#pragma once

#include <RadFiled3D/nn/exception.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

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

}  // namespace RadFiled3D::nn::memory
