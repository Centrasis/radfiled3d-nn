// The C ABI.
//
// Three rules hold everywhere below, and they are what makes the boundary safe:
//
//   * Nothing throws across it. Every entry point runs inside `guard`; an exception becomes a
//     status code, never a stack unwinding into a foreign runtime.
//   * No decision is made here. The C ABI is a projection of the C++ API, not a second
//     implementation (R-X3). It owns handle bookkeeping and nothing else — a second copy of the
//     rules would be a second thing to get wrong, which is exactly defect D2.
//   * Errors carry a sentence. The status code says what kind of failure it was;
//     `rfnn_last_error` returns the message that was actually produced.
#define RFNN_BUILDING
#include <RadFiled3D/nn/c_api.h>

#include <RadFiled3D/nn/backends/compute_backend.hpp>
#include <RadFiled3D/nn/backends/onnx.hpp>
#include <RadFiled3D/nn/core/session.hpp>
#include <RadFiled3D/nn/deploy/package.hpp>
#include <RadFiled3D/nn/exception.hpp>
#include <RadFiled3D/nn/memory/dx11.hpp>
#include <RadFiled3D/nn/memory/dx12.hpp>
#include <RadFiled3D/nn/memory/memory_ref.hpp>
#include <RadFiled3D/nn/memory/vk.hpp>
#include <RadFiled3D/nn/version.hpp>

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace {

// A shorthand, not a `using namespace`: this file defines extern "C" symbols and must not pull a
// namespace into scope around them.
namespace nn = RadFiled3D::nn;

thread_local std::string g_last_error;

rfnn_status status_of(const RadFiled3D::nn::Exception& err) {
    switch (err.get_kind()) {
        case RadFiled3D::nn::ErrorKind::FeatureDisabled: return RFNN_FEATURE_DISABLED;
        case RadFiled3D::nn::ErrorKind::NotFound: return RFNN_NOT_FOUND;
        case RadFiled3D::nn::ErrorKind::Io: return RFNN_IO;
        case RadFiled3D::nn::ErrorKind::InvalidArgument:
        case RadFiled3D::nn::ErrorKind::UnsupportedInterop: return RFNN_INVALID_ARGUMENT;
        default: return RFNN_BAD_PACKAGE;
    }
}

rfnn_status fail(rfnn_status status, std::string message) {
    g_last_error = std::move(message);
    return status;
}

/// Run `f`, turning any exception into a status code rather than letting it unwind into C.
template <typename F>
rfnn_status guard(F&& f) noexcept {
    try {
        return f();
    } catch (const RadFiled3D::nn::Exception& err) {
        return fail(status_of(err), err.what());
    } catch (const std::exception& err) {
        return fail(RFNN_INTERNAL, std::string("internal error: ") + err.what());
    } catch (...) {
        return fail(RFNN_INTERNAL, "internal error: unknown exception at the C ABI boundary");
    }
}

}  // namespace

/// The handle keeps the decoded header; tensor names are `std::string`s, which are already
/// NUL-terminated, so the accessors hand out pointers into them that stay valid until free.
///
/// The semantic and normalizer names are cached beside it for the same reason: a `Semantic` answers
/// with a `string_view` into a static table or into itself, and neither is a thing a C caller can
/// hold on to. Materialising them once, here, is the only state this ABI owns.
struct rfnn_metadata {
    explicit rfnn_metadata(RadFiled3D::nn::deploy::Metadata metadata) : inner(std::move(metadata)) {
        for (const auto& tensor : inner.io) {
            semantics.emplace_back(tensor.semantic.get_name());
            normalizers.emplace_back(RadFiled3D::nn::deploy::get_name(tensor.normalizer));
        }
    }
    RadFiled3D::nn::deploy::Metadata inner;
    std::vector<std::string> semantics;
    std::vector<std::string> normalizers;
};

/// A bindable buffer. Owns a REFERENCE, never the allocation behind it — so freeing this handle
/// releases an import, and never the engine's memory.
struct rfnn_memory {
    std::shared_ptr<RadFiled3D::nn::memory::MemoryRef> ref;
};

/// A loaded model. Holds its own reference to everything bound to it, which is what lets a caller
/// free an `rfnn_memory` the moment it has been bound.
struct rfnn_session {
    std::unique_ptr<RadFiled3D::nn::InferenceSession> inner;
};

extern "C" {

const char* rfnn_last_error(void) { return g_last_error.empty() ? nullptr : g_last_error.c_str(); }

const char* rfnn_version(void) { return RFNN_VERSION; }

rfnn_status rfnn_metadata_read(const char* path, rfnn_metadata** out) {
    return guard([&]() -> rfnn_status {
        if (!path || !out) return fail(RFNN_INVALID_ARGUMENT, "rfnn_metadata_read: null argument");
        *out = nullptr;
        auto* handle = new rfnn_metadata(RadFiled3D::nn::deploy::Package::read_metadata_file(path));
        *out = handle;
        return RFNN_OK;
    });
}

void rfnn_metadata_free(rfnn_metadata* handle) { delete handle; }

uint32_t rfnn_metadata_tensor_count(const rfnn_metadata* handle) {
    return handle ? static_cast<uint32_t>(handle->inner.io.size()) : 0u;
}

const char* rfnn_metadata_tensor_name(const rfnn_metadata* handle, uint32_t index) {
    if (!handle || index >= handle->inner.io.size()) return nullptr;
    return handle->inner.io[index].name.c_str();
}

int32_t rfnn_metadata_tensor_is_output(const rfnn_metadata* handle, uint32_t index) {
    if (!handle || index >= handle->inner.io.size()) return -1;
    return handle->inner.io[index].role == RadFiled3D::nn::deploy::Role::Output ? 1 : 0;
}

}  // extern "C"

namespace {

/// Build an ExternalBuffer from the flat arguments a C caller can express.
nn::memory::ExternalBuffer external_buffer(std::uint64_t size_bytes, std::uint64_t offset_bytes,
                                           std::uint64_t region_bytes, const std::uint8_t* device_uuid,
                                           bool dedicated) {
    nn::memory::ExternalBuffer buffer;
    buffer.size_bytes = size_bytes;
    buffer.offset_bytes = offset_bytes;
    buffer.region_bytes = region_bytes;
    buffer.dedicated = dedicated;
    // All-zero means "the current device", so a null pointer is the same request spelled shorter.
    if (device_uuid != nullptr) std::copy_n(device_uuid, buffer.device_uuid.size(), buffer.device_uuid.begin());
    return buffer;
}

/// Import through whichever graphics backend owns `domain`. The C entry points differ only in how
/// they name a handle; everything after that is one path.
rfnn_status import_into(nn::memory::ExternalMemory& interop, const nn::memory::ExternalBuffer& buffer,
                        const char* compute_backend, rfnn_memory** out) {
    if (out == nullptr) return fail(RFNN_INVALID_ARGUMENT, "import: null out parameter");
    *out = nullptr;
    if (compute_backend == nullptr) return fail(RFNN_INVALID_ARGUMENT, "import: null backend name");
    // The vocabulary is the C++ one; this ABI forwards a name and re-decides nothing (R-I3).
    const nn::Backend backend = nn::backend_from_name(compute_backend);
    *out = new rfnn_memory{interop.import_buffer(buffer, backend)};
    return RFNN_OK;
}

const nn::deploy::TensorDescriptor* tensor_at(const rfnn_metadata* handle, std::uint32_t index) {
    if (handle == nullptr || index >= handle->inner.io.size()) return nullptr;
    return &handle->inner.io[index];
}

}  // namespace

extern "C" {

// ── metadata: what a tensor declares ─────────────────────────────────────────────────────────────

const char* rfnn_metadata_tensor_semantic(const rfnn_metadata* handle, uint32_t index) {
    if (handle == nullptr || index >= handle->semantics.size()) return nullptr;
    return handle->semantics[index].c_str();
}

const char* rfnn_metadata_tensor_unit(const rfnn_metadata* handle, uint32_t index) {
    const auto* tensor = tensor_at(handle, index);
    return tensor == nullptr ? nullptr : tensor->unit.c_str();
}

const char* rfnn_metadata_tensor_normalizer(const rfnn_metadata* handle, uint32_t index) {
    if (handle == nullptr || index >= handle->normalizers.size()) return nullptr;
    return handle->normalizers[index].c_str();
}

int32_t rfnn_metadata_tensor_rank(const rfnn_metadata* handle, uint32_t index) {
    const auto* tensor = tensor_at(handle, index);
    return tensor == nullptr ? -1 : static_cast<int32_t>(tensor->shape.size());
}

int32_t rfnn_metadata_tensor_shape(const rfnn_metadata* handle, uint32_t index, uint32_t* shape,
                                   uint32_t capacity) {
    const auto* tensor = tensor_at(handle, index);
    if (tensor == nullptr) return -1;
    const auto count = static_cast<uint32_t>(tensor->shape.size());
    if (shape == nullptr) return static_cast<int32_t>(count);
    const uint32_t written = capacity < count ? capacity : count;
    std::copy_n(tensor->shape.begin(), written, shape);
    return static_cast<int32_t>(written);
}

int32_t rfnn_metadata_tensor_range(const rfnn_metadata* handle, uint32_t index, double* min, double* max) {
    const auto* tensor = tensor_at(handle, index);
    if (tensor == nullptr) return -1;
    // Only a plain interval is expressible here. A histogram or a categorical range says 0 rather
    // than flattening into a min/max that would read as something the package never claimed.
    if (const auto* interval = std::get_if<nn::deploy::MinMax>(&tensor->range)) {
        if (min != nullptr) *min = interval->min;
        if (max != nullptr) *max = interval->max;
        return 1;
    }
    return 0;
}

// ── memory ───────────────────────────────────────────────────────────────────────────────────────

rfnn_status rfnn_memory_from_host(void* data, uint64_t size_bytes, rfnn_memory** out) {
    return guard([&]() -> rfnn_status {
        if (out == nullptr) return fail(RFNN_INVALID_ARGUMENT, "rfnn_memory_from_host: null out parameter");
        *out = nullptr;
        if (data == nullptr && size_bytes != 0)
            return fail(RFNN_INVALID_ARGUMENT, "rfnn_memory_from_host: null pointer with a non-zero size");
        *out = new rfnn_memory{std::make_shared<nn::memory::host::MemoryRef>(data, size_bytes)};
        return RFNN_OK;
    });
}

rfnn_status rfnn_memory_import_vulkan_fd(int fd, uint64_t size_bytes, uint64_t offset_bytes,
                                         uint64_t region_bytes, const uint8_t device_uuid[16],
                                         int32_t dedicated, const char* compute_backend,
                                         rfnn_memory** out) {
    return guard([&]() -> rfnn_status {
        auto buffer = external_buffer(size_bytes, offset_bytes, region_bytes, device_uuid, dedicated != 0);
        buffer.handle = nn::memory::OpaqueFd{fd};
        nn::memory::vk::ExternalMemory interop;
        return import_into(interop, buffer, compute_backend, out);
    });
}

rfnn_status rfnn_memory_import_vulkan_win32(void* handle, uint64_t size_bytes, uint64_t offset_bytes,
                                            uint64_t region_bytes, const uint8_t device_uuid[16],
                                            int32_t dedicated, const char* compute_backend,
                                            rfnn_memory** out) {
    return guard([&]() -> rfnn_status {
        auto buffer = external_buffer(size_bytes, offset_bytes, region_bytes, device_uuid, dedicated != 0);
        buffer.handle = nn::memory::Win32Handle{handle};
        nn::memory::vk::ExternalMemory interop;
        return import_into(interop, buffer, compute_backend, out);
    });
}

rfnn_status rfnn_memory_import_d3d12(void* shared_handle, rfnn_d3d12_kind kind, uint64_t size_bytes,
                                     uint64_t offset_bytes, uint64_t region_bytes,
                                     const uint8_t device_uuid[16], const char* compute_backend,
                                     rfnn_memory** out) {
    return guard([&]() -> rfnn_status {
        // `dedicated` is not a parameter here on purpose: CUDA dictates it from the kind, so letting
        // a caller pass it would only let them be wrong.
        auto buffer = external_buffer(size_bytes, offset_bytes, region_bytes, device_uuid, false);
        buffer.handle = nn::memory::D3D12Handle{
            shared_handle, kind == RFNN_D3D12_HEAP ? nn::memory::D3D12Handle::Kind::Heap
                                                   : nn::memory::D3D12Handle::Kind::Resource};
        nn::memory::dx12::ExternalMemory interop;
        return import_into(interop, buffer, compute_backend, out);
    });
}

rfnn_status rfnn_memory_import_d3d11(void* shared_handle, rfnn_d3d11_kind kind, uint64_t size_bytes,
                                     uint64_t offset_bytes, uint64_t region_bytes,
                                     const uint8_t device_uuid[16], const char* compute_backend,
                                     rfnn_memory** out) {
    return guard([&]() -> rfnn_status {
        auto buffer = external_buffer(size_bytes, offset_bytes, region_bytes, device_uuid, false);
        buffer.handle = nn::memory::D3D11Handle{
            shared_handle, kind == RFNN_D3D11_KMT ? nn::memory::D3D11Handle::Kind::KmtHandle
                                                  : nn::memory::D3D11Handle::Kind::NtHandle};
        nn::memory::dx11::ExternalMemory interop;
        return import_into(interop, buffer, compute_backend, out);
    });
}

rfnn_status rfnn_memory_from_d3d12_resource(void* resource, uint64_t size_bytes, uint64_t offset_bytes,
                                            rfnn_memory** out) {
    return guard([&]() -> rfnn_status {
        if (out == nullptr)
            return fail(RFNN_INVALID_ARGUMENT, "rfnn_memory_from_d3d12_resource: null out parameter");
        *out = nullptr;
        auto buffer = external_buffer(size_bytes, offset_bytes, size_bytes, nullptr, false);
        buffer.handle = nn::memory::D3D12NativeResource{resource};
        // Straight to the DirectML backend: this is the case where nothing is imported, so it does
        // not go through a graphics interop at all.
        *out = new rfnn_memory{
            nn::get_compute_backend(nn::Backend::DirectMl).import_external_memory(buffer)};
        return RFNN_OK;
    });
}

void rfnn_memory_free(rfnn_memory* handle) { delete handle; }

uint64_t rfnn_memory_size_bytes(const rfnn_memory* handle) {
    return handle == nullptr || !handle->ref ? 0u : handle->ref->get_size_bytes();
}

int32_t rfnn_memory_domain(const rfnn_memory* handle) {
    if (handle == nullptr || !handle->ref) return -1;
    return static_cast<int32_t>(handle->ref->get_domain());
}

// ── inference ────────────────────────────────────────────────────────────────────────────────────

rfnn_status rfnn_session_load(const char* path, const char* backend, int32_t device,
                              rfnn_session** out) {
    return guard([&]() -> rfnn_status {
        if (path == nullptr || backend == nullptr || out == nullptr)
            return fail(RFNN_INVALID_ARGUMENT, "rfnn_session_load: null argument");
        *out = nullptr;
        const nn::Backend selected = nn::backend_from_name(backend);
        auto session = nn::onnx::load(nn::deploy::Package::read_file(path), selected, device);
        *out = new rfnn_session{std::move(session)};
        return RFNN_OK;
    });
}

void rfnn_session_free(rfnn_session* handle) { delete handle; }

rfnn_status rfnn_session_set_voxel_grid(rfnn_session* handle, uint32_t x, uint32_t y, uint32_t z) {
    return guard([&]() -> rfnn_status {
        if (handle == nullptr) return fail(RFNN_INVALID_ARGUMENT, "rfnn_session_set_voxel_grid: null session");
        handle->inner->set_voxel_grid({x, y, z});
        return RFNN_OK;
    });
}

namespace {
rfnn_status bind(rfnn_session* handle, const char* name, rfnn_memory* memory, bool is_input) {
    if (handle == nullptr || name == nullptr || memory == nullptr || !memory->ref)
        return fail(RFNN_INVALID_ARGUMENT, "bind: null argument");
    // The session takes its own reference, so the caller may free the handle straight after.
    if (is_input)
        handle->inner->bind_input(name, memory->ref);
    else
        handle->inner->bind_output(name, memory->ref);
    return RFNN_OK;
}
}  // namespace

rfnn_status rfnn_session_bind_input(rfnn_session* handle, const char* name, rfnn_memory* memory) {
    return guard([&] { return bind(handle, name, memory, true); });
}

rfnn_status rfnn_session_bind_output(rfnn_session* handle, const char* name, rfnn_memory* memory) {
    return guard([&] { return bind(handle, name, memory, false); });
}

rfnn_status rfnn_session_infer(rfnn_session* handle) {
    return guard([&]() -> rfnn_status {
        if (handle == nullptr) return fail(RFNN_INVALID_ARGUMENT, "rfnn_session_infer: null session");
        handle->inner->infer();
        return RFNN_OK;
    });
}

const char* rfnn_session_backend(const rfnn_session* handle) {
    if (handle == nullptr) return nullptr;
    // A static string from the vocabulary table, so it needs no caching and no freeing.
    const std::string_view name = nn::to_string(handle->inner->get_backend());
    return name.data();
}

}  // extern "C"
