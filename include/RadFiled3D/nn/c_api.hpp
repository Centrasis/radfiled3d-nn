// Header-only C++20 RAII wrapper over the C ABI.
//
// This is what an engine plugin includes when it links a PREBUILT radfiled3d-nn: handles become
// move-only classes freed on destruction, status codes become exceptions carrying the message the
// library actually produced, and index/count pairs become ranges. It adds no logic of its own —
// every decision stays inside the library (R-X3) — and it pulls in nothing from <RadFiled3D/nn.hpp>,
// so it is safe across a toolchain boundary. A consumer building from source with its own
// toolchain can use the C++ API directly instead.
#ifndef RFNN_C_API_HPP
#define RFNN_C_API_HPP

#include <RadFiled3D/nn/c_api.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace RadFiled3D::nn::c {

class Exception : public std::runtime_error {
public:
    Exception(rfnn_status status, std::string message) : std::runtime_error(std::move(message)), status_(status) {}
    rfnn_status status() const noexcept { return status_; }

private:
    rfnn_status status_;
};

namespace detail {
inline void check(rfnn_status status, const std::string& what) {
    if (status == RFNN_OK) return;
    const char* message = rfnn_last_error();
    throw Exception(status, what + ": " + (message ? message : "unknown error"));
}
}  // namespace detail

/// What a package declares, without loading a runtime.
class Metadata {
public:
    explicit Metadata(const std::string& path) {
        detail::check(rfnn_metadata_read(path.c_str(), &handle_), "reading " + path);
    }
    ~Metadata() { rfnn_metadata_free(handle_); }

    Metadata(const Metadata&) = delete;
    Metadata& operator=(const Metadata&) = delete;
    Metadata(Metadata&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    Metadata& operator=(Metadata&& other) noexcept {
        if (this != &other) {
            rfnn_metadata_free(handle_);
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    std::uint32_t tensor_count() const { return rfnn_metadata_tensor_count(handle_); }

    std::string tensor_name(std::uint32_t index) const {
        const char* name = rfnn_metadata_tensor_name(handle_, index);
        if (!name) throw Exception(RFNN_NOT_FOUND, "tensor index out of range");
        return name;
    }

    bool tensor_is_output(std::uint32_t index) const {
        const std::int32_t is_output = rfnn_metadata_tensor_is_output(handle_, index);
        if (is_output < 0) throw Exception(RFNN_NOT_FOUND, "tensor index out of range");
        return is_output != 0;
    }

    std::vector<std::string> input_names() const { return names_where(false); }
    std::vector<std::string> output_names() const { return names_where(true); }

private:
    std::vector<std::string> names_where(bool outputs) const {
        std::vector<std::string> out;
        for (std::uint32_t i = 0; i < tensor_count(); ++i)
            if (tensor_is_output(i) == outputs) out.push_back(tensor_name(i));
        return out;
    }

    rfnn_metadata* handle_ = nullptr;
};

/// A bindable buffer. Move-only, released on destruction — the mapping goes with it, the engine's
/// allocation does not.
class Memory {
public:
    /// Ordinary process memory the caller owns.
    static Memory host(void* data, std::uint64_t size_bytes) {
        rfnn_memory* handle = nullptr;
        detail::check(rfnn_memory_from_host(data, size_bytes, &handle), "wrapping host memory");
        return Memory(handle);
    }

    /// A Vulkan allocation exported with `vkGetMemoryFdKHR`. **The descriptor is consumed** on
    /// success — do not close it.
    static Memory import_vulkan(int fd, std::uint64_t size_bytes, std::uint64_t offset_bytes,
                                std::uint64_t region_bytes, const std::uint8_t device_uuid[16],
                                bool dedicated, const std::string& compute_backend) {
        rfnn_memory* handle = nullptr;
        detail::check(rfnn_memory_import_vulkan_fd(fd, size_bytes, offset_bytes, region_bytes,
                                                   device_uuid, dedicated ? 1 : 0,
                                                   compute_backend.c_str(), &handle),
                      "importing a Vulkan allocation");
        return Memory(handle);
    }

    /// A D3D12 object shared with `ID3D12Device::CreateSharedHandle`.
    static Memory import_d3d12(void* shared_handle, rfnn_d3d12_kind kind, std::uint64_t size_bytes,
                               std::uint64_t offset_bytes, std::uint64_t region_bytes,
                               const std::uint8_t device_uuid[16], const std::string& compute_backend) {
        rfnn_memory* handle = nullptr;
        detail::check(rfnn_memory_import_d3d12(shared_handle, kind, size_bytes, offset_bytes,
                                               region_bytes, device_uuid, compute_backend.c_str(),
                                               &handle),
                      "importing a D3D12 resource");
        return Memory(handle);
    }

    /// The `ID3D12Resource` itself, for a DirectML session. Nothing is imported.
    static Memory from_d3d12_resource(void* resource, std::uint64_t size_bytes,
                                      std::uint64_t offset_bytes = 0) {
        rfnn_memory* handle = nullptr;
        detail::check(rfnn_memory_from_d3d12_resource(resource, size_bytes, offset_bytes, &handle),
                      "wrapping a D3D12 resource");
        return Memory(handle);
    }

    ~Memory() { rfnn_memory_free(handle_); }
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
    Memory(Memory&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    Memory& operator=(Memory&& other) noexcept {
        if (this != &other) {
            rfnn_memory_free(handle_);
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    std::uint64_t size_bytes() const { return rfnn_memory_size_bytes(handle_); }
    rfnn_domain domain() const { return static_cast<rfnn_domain>(rfnn_memory_domain(handle_)); }
    rfnn_memory* get() const noexcept { return handle_; }

private:
    explicit Memory(rfnn_memory* handle) : handle_(handle) {}
    rfnn_memory* handle_ = nullptr;
};

/// A loaded model, driven through the protocol in c_api.h:
/// `set_voxel_grid` → `bind_input`/`bind_output` → `infer`.
class Session {
public:
    Session(const std::string& path, const std::string& backend, std::int32_t device = -1) {
        detail::check(rfnn_session_load(path.c_str(), backend.c_str(), device, &handle_),
                      "loading " + path + " on " + backend);
    }

    ~Session() { rfnn_session_free(handle_); }
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    Session& operator=(Session&& other) noexcept {
        if (this != &other) {
            rfnn_session_free(handle_);
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    void set_voxel_grid(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
        detail::check(rfnn_session_set_voxel_grid(handle_, x, y, z), "setting the voxel grid");
    }
    /// The session keeps its own reference, so `memory` may be destroyed straight after.
    void bind_input(const std::string& name, const Memory& memory) {
        detail::check(rfnn_session_bind_input(handle_, name.c_str(), memory.get()),
                      "binding input " + name);
    }
    void bind_output(const std::string& name, const Memory& memory) {
        detail::check(rfnn_session_bind_output(handle_, name.c_str(), memory.get()),
                      "binding output " + name);
    }
    void infer() { detail::check(rfnn_session_infer(handle_), "running inference"); }

    std::string backend() const {
        const char* name = rfnn_session_backend(handle_);
        return name == nullptr ? std::string{} : name;
    }

private:
    rfnn_session* handle_ = nullptr;
};

inline std::string version() { return rfnn_version(); }

}  // namespace RadFiled3D::nn::c

#endif  // RFNN_C_API_HPP
