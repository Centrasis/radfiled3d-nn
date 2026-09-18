// `ComputeBackend::adopt` — the one call that makes a reference usable by a backend.
//
// Every bind path runs its input through this, so no caller has to know what it is holding. The
// interesting case is the third view: memory a renderer allocated, already imported once, and now
// wanted by another API. That is why `MemoryRef` carries an `Origin` at all — see memory_ref.hpp.
//
// THE REGISTRY exists so that asking twice is free and, more importantly, CORRECT. Importing the
// same external allocation twice yields two independent mappings of one piece of memory: two
// address ranges, two `cudaDestroyExternalMemory` obligations, and two references that compare
// unequal while aliasing the same bytes. Handing back the reference that already exists avoids all
// of it. Entries are `weak_ptr`, so the cache never keeps a mapping alive: when the last caller
// drops its reference the import is released and the next `adopt` genuinely re-imports.
#include <RadFiled3D/nn/backends/compute_backend.hpp>

#include <map>
#include <mutex>

namespace RadFiled3D::nn {

namespace {

/// What makes two adoptions the same adoption.
///
/// The handle VALUE alone is not enough: a POSIX descriptor 7 and an NT handle 0x7 are different
/// things, so the variant's index is part of the key. The offset is too, because a renderer
/// suballocates and two regions of one heap are two different buffers.
struct AdoptKey {
    std::size_t handle_kind = 0;
    /// `D3D12Handle::Kind` / `D3D11Handle::Kind`. A committed resource and a heap take DIFFERENT
    /// CUDA import types, so two otherwise identical handles are not the same adoption.
    unsigned sub_kind = 0;
    std::uint64_t handle = 0;
    std::uint64_t offset = 0;
    memory::Domain target = memory::Domain::Host;
    int device = -1;

    auto operator<=>(const AdoptKey&) const = default;
};

/// The native handle reduced to a comparable identity: which alternative, which sub-kind, what value.
AdoptKey identify(const memory::ExternalHandle& handle) {
    AdoptKey key;
    key.handle_kind = handle.index();
    std::visit(
        [&](const auto& h) {
            using T = std::decay_t<decltype(h)>;
            if constexpr (std::is_same_v<T, memory::OpaqueFd>) {
                key.handle = static_cast<std::uint64_t>(h.fd);
            } else if constexpr (std::is_same_v<T, memory::D3D12NativeResource>) {
                key.handle = reinterpret_cast<std::uintptr_t>(h.resource);
            } else if constexpr (std::is_same_v<T, memory::Win32Handle>) {
                key.handle = reinterpret_cast<std::uintptr_t>(h.handle);
            } else {
                key.handle = reinterpret_cast<std::uintptr_t>(h.handle);
                key.sub_kind = static_cast<unsigned>(h.kind);
            }
        },
        handle);
    return key;
}

std::map<AdoptKey, std::weak_ptr<memory::MemoryRef>>& registry() {
    static std::map<AdoptKey, std::weak_ptr<memory::MemoryRef>> map;
    return map;
}
std::mutex& registry_lock() {
    static std::mutex lock;
    return lock;
}

}  // namespace

std::shared_ptr<memory::MemoryRef> ComputeBackend::adopt(std::shared_ptr<memory::MemoryRef> memory,
                                                         int device) const {
    if (!memory) throw Exception::invalid_argument("no memory to adopt");

    // ── already ours ────────────────────────────────────────────────────────────────────────
    if (memory->get_domain() == get_memory_domain()) {
        const int on = memory->get_device_index();
        // -1 is "no such notion", never a synonym for device 0, so it is not a mismatch.
        if (on >= 0 && device >= 0 && on != device)
            throw Exception::invalid_argument(
                std::string("memory is on ") + std::string(memory::to_string(memory->get_domain())) +
                " device " + std::to_string(on) + " but this session runs on device " +
                std::to_string(device) + "; load the session on the device the memory is already on");
        return memory;
    }

    // ── host memory binds to ANY backend, and always could ──────────────────────────────────
    // Not a fallback this function invents: a runtime takes host memory with CPU memory info and
    // does the transfer itself (`memory_info_for` has handled `Domain::Host` first since before
    // this function existed, and `FieldInference` binds a host span to a CUDA session in the
    // tests). Refusing it here would break the ordinary way a caller passes positions in.
    if (memory->get_domain() == memory::Domain::Host) return memory;

    // ── a second view of memory some other API owns ─────────────────────────────────────────
    const auto* external = std::get_if<memory::ExternalOrigin>(&memory->get_origin());
    if (external == nullptr)
        throw Exception::invalid_argument(
            std::string("cannot adopt ") + std::string(memory::to_string(memory->get_domain())) +
            " memory into a " + std::string(to_string(get_backend())) +
            " session: it records no exported handle, so there is nothing to import. Export the "
            "allocation from the API that owns it and import that.");

    // An IMPOSSIBLE pairing and one that is merely not compiled in are different errors, and
    // conflating them sends someone to fix the wrong thing: Vulkan memory can never reach the CPU
    // provider, while Vulkan memory reaching CUDA is a matter of `RFNN_WITH_CUDA`. The first is
    // answered here; the second by `import_external_memory` below, which names its own option.
    if (!can_import_from(memory->get_domain()))
        throw Exception::unsupported_interop(std::string(memory::to_string(memory->get_domain())),
                                             std::string(to_string(get_backend())));

    AdoptKey key = identify(external->handle);
    key.offset = external->offset_bytes;
    key.target = get_memory_domain();
    key.device = device;

    std::lock_guard<std::mutex> guard(registry_lock());

    // A lookup with no device asked for matches ANY device this allocation is already on. The
    // caller that says -1 means "wherever it lands"; refusing to match an entry made under the
    // resolved ordinal would re-import — and for a descriptor CUDA has already consumed, that
    // fails in the driver rather than anywhere explicable.
    for (auto it = registry().begin(); it != registry().end();) {
        const bool same_allocation = it->first.handle_kind == key.handle_kind &&
                                     it->first.sub_kind == key.sub_kind &&
                                     it->first.handle == key.handle &&
                                     it->first.offset == key.offset && it->first.target == key.target;
        if (!same_allocation || (device >= 0 && it->first.device != device)) {
            ++it;
            continue;
        }
        if (auto existing = it->second.lock()) return existing;
        it = registry().erase(it);  // the last holder let go; the mapping went with it
    }

    // Only NOW does a spent handle matter. Checked after the lookup, because an allocation that
    // was already imported is served from the registry and never needs its handle again — the
    // descriptor being gone is irrelevant until something actually has to import.
    if (external->consumed)
        throw Exception::invalid_argument(
            "the handle this memory was imported from has been consumed — CUDA closes a POSIX file "
            "descriptor on import, so it cannot be imported a second time. Export another handle "
            "for this allocation. (An NT handle is duplicated instead and does not have this "
            "restriction.)");

    auto imported = import_external_memory(*external);
    // Keyed on the RESOLVED device, not the requested one: the import resolves it from the
    // allocation's UUID, so -1 becomes a real ordinal here and a later lookup naming that ordinal
    // has to find this entry.
    key.device = imported->get_device_index();
    registry().emplace(key, imported);
    // The source handle is spent if the platform consumes it. Told to the reference that holds it,
    // because only it can answer the next `adopt` accurately.
    if (std::holds_alternative<memory::OpaqueFd>(external->handle)) memory->mark_origin_consumed();
    return imported;
}

}  // namespace RadFiled3D::nn
