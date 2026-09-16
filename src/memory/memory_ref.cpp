#include <RadFiled3D/nn/memory/memory_ref.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace RadFiled3D::nn::memory {

std::string_view to_string(Domain domain) noexcept {
    switch (domain) {
        case Domain::Host: return "host";
        case Domain::Cuda: return "cuda";
        case Domain::Hip: return "hip";
        case Domain::Vulkan: return "vulkan";
        case Domain::D3D11: return "d3d11";
        case Domain::D3D12: return "d3d12";
    }
    return "?";
}

std::uint64_t MemoryRef::get_element_count(std::uint64_t element_bytes) const {
    if (element_bytes == 0) throw Exception::invalid_argument("MemoryRef: element size of zero bytes");
    return get_size_bytes() / element_bytes;
}

void MemoryRef::require_domain(Domain expected) const {
    if (get_domain() != expected)
        throw Exception::invalid_argument("expected " + std::string(to_string(expected)) +
                                          " memory, got " + std::string(to_string(get_domain())));
}

namespace host {

namespace {

/// A host allocation that owns its storage. Not in the header: nothing outside needs the type, only
/// the `MemoryRef` it is.
class OwnedMemoryRef final : public memory::MemoryRef {
public:
    explicit OwnedMemoryRef(std::uint64_t bytes) : storage_(static_cast<std::size_t>(bytes)) {}

    Domain get_domain() const noexcept override { return Domain::Host; }
    std::uint64_t get_size_bytes() const noexcept override { return storage_.size(); }
    std::uint64_t get_address() const noexcept override {
        return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(storage_.data()));
    }

private:
    std::vector<std::byte> storage_;
};

}  // namespace

std::shared_ptr<memory::MemoryRef> allocate(std::uint64_t bytes) {
    return std::make_shared<OwnedMemoryRef>(bytes);
}

}  // namespace host

}  // namespace RadFiled3D::nn::memory
