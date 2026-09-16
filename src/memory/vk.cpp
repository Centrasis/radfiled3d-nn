#include <RadFiled3D/nn/memory/vk.hpp>

#include <RadFiled3D/nn/backends/vulkan.hpp>
#include <RadFiled3D/nn/backends/compute_backend.hpp>

namespace RadFiled3D::nn::memory::vk {

// This file names no compute backend, and that is the point (rule 7). Which backends can consume
// a Vulkan allocation is the COMPUTE side's knowledge — it owns the external-memory API that does the
// importing — so it is asked, never switched on. Adding a backend adds its files and leaves this
// one untouched.

bool ExternalMemory::supports(Backend compute) const noexcept {
    if (!vulkan::available()) return false;
    const ComputeBackend& backend = get_compute_backend(compute);
    return backend.available() && backend.can_import_from(get_domain());
}

std::shared_ptr<memory::MemoryRef> ExternalMemory::import_buffer(const ExternalBuffer& buffer,
                                                                 Backend compute) {
    const ComputeBackend& backend = get_compute_backend(compute);
    // An IMPOSSIBLE pairing is not a configuration mistake, so it gets its own error: no build
    // option would ever make it work (R-G3).
    if (!backend.can_import_from(get_domain()))
        throw Exception::unsupported_interop(to_string(get_domain()), to_string(compute));
    // Compiled out, on either side: name the option that would enable it.
    if (!vulkan::available())
        throw Exception::feature_disabled("vulkan", "importing a Vulkan allocation as an inference target");
    if (!backend.available())
        throw Exception::feature_disabled(backend.get_feature(), "importing a Vulkan allocation as an inference target");
    return backend.import_external_memory(buffer);
}

ExternalBuffer ExternalMemory::export_buffer(std::uint64_t, Backend compute) {
    const ComputeBackend& backend = get_compute_backend(compute);
    if (!backend.can_import_from(get_domain()))
        throw Exception::unsupported_interop(to_string(get_domain()), to_string(compute));
    if (!vulkan::available())
        throw Exception::feature_disabled("vulkan", "exporting a device buffer to Vulkan");
    throw Exception::feature_disabled(backend.get_feature(), "exporting a device buffer to Vulkan");
}

}  // namespace RadFiled3D::nn::memory::vk
