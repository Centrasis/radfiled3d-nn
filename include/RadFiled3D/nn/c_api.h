/* radfiled3d-nn — the C ABI.
 *
 * The engine-facing surface (requirements.md R-X1). A prebuilt binary of this library is consumed
 * from an Unreal Engine plugin — a different toolchain, CRT and STL from the one that built it — so
 * the boundary a plugin links against is C: opaque handles, integer status codes, C strings. The
 * C++ API in <RadFiled3D/nn.hpp> is the same API for consumers who build from source with their own
 * toolchain; this header is a projection of it and makes no decision of its own (R-X3).
 *
 * HAND-MAINTAINED, updated in the same change as src/capi/c_api.cpp; tests/c_api.cpp drives every
 * function declared here the way C does. Prefer <RadFiled3D/nn/c_api.hpp>, the RAII wrapper, in C++ code.
 *
 * Conventions
 *   - Every function returning rfnn_status leaves a message in rfnn_last_error() on failure.
 *     The message is thread-local and valid until the next failing call on that thread.
 *   - No exception ever crosses this boundary.
 *   - Handles are freed exactly once with their _free function; freeing NULL is a no-op.
 */
#ifndef RFNN_C_API_H
#define RFNN_C_API_H

#include <stdint.h>

/* Symbol visibility. The C ABI is exported from the shared library `rfnn_c` (target rfnn::c) and
 * from nothing else. */
#if defined(_WIN32)
#  if defined(RFNN_BUILDING)
#    define RFNN_API __declspec(dllexport)
#  elif defined(RFNN_SHARED)
#    define RFNN_API __declspec(dllimport)
#  else
#    define RFNN_API
#  endif
#elif defined(RFNN_BUILDING) && defined(__GNUC__)
#  define RFNN_API __attribute__((visibility("default")))
#else
#  define RFNN_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum rfnn_status {
    RFNN_OK = 0,
    RFNN_INVALID_ARGUMENT = 1,
    RFNN_NOT_FOUND = 2,
    RFNN_BAD_PACKAGE = 3,
    RFNN_FEATURE_DISABLED = 4,
    RFNN_IO = 5,
    RFNN_INTERNAL = 6
} rfnn_status;

/* The message from the most recent failing call on this thread, or NULL. */
RFNN_API const char* rfnn_last_error(void);

/* The library version. Never fails, never needs freeing. */
RFNN_API const char* rfnn_version(void);

/* ── package metadata ────────────────────────────────────────────────────────────────────────
 * Reading metadata creates no inference session and decodes no ONNX graph, so an editor can list
 * models and show their training ranges without touching the GPU.
 */
typedef struct rfnn_metadata rfnn_metadata;

RFNN_API rfnn_status rfnn_metadata_read(const char* path, rfnn_metadata** out);
RFNN_API void        rfnn_metadata_free(rfnn_metadata* handle);

/* Declared tensors, inputs and outputs together. 0 for a NULL handle. */
RFNN_API uint32_t    rfnn_metadata_tensor_count(const rfnn_metadata* handle);
/* Name of tensor `index`, or NULL when out of range. Valid until the handle is freed. */
RFNN_API const char* rfnn_metadata_tensor_name(const rfnn_metadata* handle, uint32_t index);
/* 1 = output, 0 = input, -1 = invalid index or NULL handle. */
RFNN_API int32_t     rfnn_metadata_tensor_is_output(const rfnn_metadata* handle, uint32_t index);

/* What a tensor MEANS and what it is measured in. Both are names on the wire, so a semantic this
 * build has never heard of still comes back as its own string rather than as "unknown" — that is
 * the whole point of the descriptor design (R-F2). NULL for an invalid index; valid until free. */
RFNN_API const char* rfnn_metadata_tensor_semantic(const rfnn_metadata* handle, uint32_t index);
RFNN_API const char* rfnn_metadata_tensor_unit(const rfnn_metadata* handle, uint32_t index);
/* How a metric value reaches the graph: "identity", "linear0_1", "log_scale", ... */
RFNN_API const char* rfnn_metadata_tensor_normalizer(const rfnn_metadata* handle, uint32_t index);

/* The per-query shape. `rank` is how many dimensions there are; `shape` copies up to `capacity` of
 * them and returns how many were written, or -1 on an invalid index. Resolution lives HERE — a
 * spectrum's bin count is the trailing dimension, not a separate field. */
RFNN_API int32_t     rfnn_metadata_tensor_rank(const rfnn_metadata* handle, uint32_t index);
RFNN_API int32_t     rfnn_metadata_tensor_shape(const rfnn_metadata* handle, uint32_t index,
                                                uint32_t* shape, uint32_t capacity);

/* The interval the model was TRAINED on, so a caller can tell an in-distribution query from an
 * extrapolation. 1 when the tensor records a min/max, 0 when it records none or a kind this
 * accessor cannot express, -1 on an invalid index. */
RFNN_API int32_t     rfnn_metadata_tensor_range(const rfnn_metadata* handle, uint32_t index,
                                                double* min, double* max);

/* ── memory ──────────────────────────────────────────────────────────────────────────────────
 * A buffer the library can bind, whoever allocated it.
 *
 * The C++ side carries this as a variant of handle types; a C ABI cannot, so each way of naming
 * memory gets its own constructor and they all produce the same opaque handle. That keeps the
 * variant — and every graphics type — out of the ABI, which is what lets a plugin built with a
 * different toolchain link this at all.
 *
 * `compute_backend` names who will USE the memory ("cuda", "tensorrt", ...): importing is that
 * backend's operation, and the result is memory in that backend's domain. Passing a backend that
 * cannot reach the graphics API gives RFNN_INVALID_ARGUMENT, never a silent copy.
 *
 * A handle may be freed as soon as it is bound: the session keeps its own reference, so the
 * underlying mapping outlives the handle. What it does NOT outlive is the allocation itself — the
 * engine still owns that and must keep it alive.
 */
typedef struct rfnn_memory rfnn_memory;

typedef enum rfnn_domain {
    RFNN_DOMAIN_HOST = 0,
    RFNN_DOMAIN_CUDA = 1,
    RFNN_DOMAIN_HIP = 2,
    RFNN_DOMAIN_VULKAN = 3,
    RFNN_DOMAIN_D3D11 = 4,
    RFNN_DOMAIN_D3D12 = 5
} rfnn_domain;

/* Ordinary process memory the caller owns — a std::vector, a NumPy array, a stack buffer. */
RFNN_API rfnn_status rfnn_memory_from_host(void* data, uint64_t size_bytes, rfnn_memory** out);

/* A Vulkan allocation exported with vkGetMemoryFdKHR. THE DESCRIPTOR IS CONSUMED on success: CUDA
 * takes ownership and closes it, so the caller must not. `size_bytes` is the whole allocation;
 * `offset_bytes`/`region_bytes` select the part to use (0 region = to the end). `device_uuid` is
 * VkPhysicalDeviceIDProperties::deviceUUID, so the memory and the session land on the same GPU;
 * all-zero means the current device. `dedicated` mirrors VkMemoryDedicatedAllocateInfo. */
RFNN_API rfnn_status rfnn_memory_import_vulkan_fd(int fd, uint64_t size_bytes, uint64_t offset_bytes,
                                                  uint64_t region_bytes, const uint8_t device_uuid[16],
                                                  int32_t dedicated, const char* compute_backend,
                                                  rfnn_memory** out);

/* The same from vkGetMemoryWin32HandleKHR. Unlike the descriptor above, the NT handle is DUPLICATED
 * and stays the caller's to close. */
RFNN_API rfnn_status rfnn_memory_import_vulkan_win32(void* handle, uint64_t size_bytes,
                                                     uint64_t offset_bytes, uint64_t region_bytes,
                                                     const uint8_t device_uuid[16], int32_t dedicated,
                                                     const char* compute_backend, rfnn_memory** out);

/* Which Direct3D object a shared handle names. Not interchangeable: they import as different types,
 * and a D3D12 resource must be imported as dedicated where a heap must not. */
typedef enum rfnn_d3d12_kind {
    RFNN_D3D12_RESOURCE = 0,  /* ID3D12Resource, a committed resource */
    RFNN_D3D12_HEAP     = 1   /* ID3D12Heap the renderer suballocates from */
} rfnn_d3d12_kind;

typedef enum rfnn_d3d11_kind {
    RFNN_D3D11_NT  = 0,  /* IDXGIResource1::CreateSharedHandle */
    RFNN_D3D11_KMT = 1   /* IDXGIResource::GetSharedHandle, the older global handle */
} rfnn_d3d11_kind;

/* A D3D12 object shared with ID3D12Device::CreateSharedHandle. */
RFNN_API rfnn_status rfnn_memory_import_d3d12(void* shared_handle, rfnn_d3d12_kind kind,
                                              uint64_t size_bytes, uint64_t offset_bytes,
                                              uint64_t region_bytes, const uint8_t device_uuid[16],
                                              const char* compute_backend, rfnn_memory** out);

/* A D3D11 resource shared as an NT or KMT handle. */
RFNN_API rfnn_status rfnn_memory_import_d3d11(void* shared_handle, rfnn_d3d11_kind kind,
                                              uint64_t size_bytes, uint64_t offset_bytes,
                                              uint64_t region_bytes, const uint8_t device_uuid[16],
                                              const char* compute_backend, rfnn_memory** out);

/* The ID3D12Resource ITSELF, for a session running on DirectML.
 *
 * Not an import and not a shared handle: DirectML executes on D3D12, so a resource the caller owns
 * is already that backend's memory and nothing crosses. The caller keeps the COM reference. */
RFNN_API rfnn_status rfnn_memory_from_d3d12_resource(void* resource, uint64_t size_bytes,
                                                     uint64_t offset_bytes, rfnn_memory** out);

RFNN_API void        rfnn_memory_free(rfnn_memory* handle);
/* Bytes the reference spans, 0 for NULL. */
RFNN_API uint64_t    rfnn_memory_size_bytes(const rfnn_memory* handle);
/* An rfnn_domain value, or -1 for NULL. Where the memory ended up — an imported Vulkan buffer
 * reports the COMPUTE domain it was imported into, because that is what a session binds. */
RFNN_API int32_t     rfnn_memory_domain(const rfnn_memory* handle);

/* ── inference ───────────────────────────────────────────────────────────────────────────────
 * The protocol, and the order it must be used in (R-I1):
 *
 *   load -> set_voxel_grid -> bind_input/bind_output (once each) -> infer -> infer -> ...
 *
 * The session allocates no result and copies nothing: it writes into the buffers that were bound,
 * which is what lets a prediction land in a renderer's own memory. Bindings are registered once and
 * re-read on every run, so editing an input in place needs no rebinding — but changing the grid
 * clears them, because the shapes no longer hold.
 */
typedef struct rfnn_session rfnn_session;

/* `backend` is one of "cpu", "cuda", "tensorrt", "rocm", "directml" — the same vocabulary the C++
 * API uses, forwarded verbatim; this ABI decides nothing about it. `device` is the ordinal
 * EVERYTHING runs on — the execution provider, the buffers between stages, and any kernel stage's
 * launches; -1 means the default. Bound graphics memory must be on that same device, and a buffer
 * that says otherwise is REFUSED rather than faulted on: import first
 * (`rfnn_import_*`, which resolves the renderer's device UUID), then load on the device the import
 * landed on. */
RFNN_API rfnn_status rfnn_session_load(const char* path, const char* backend, int32_t device,
                                       rfnn_session** out);
RFNN_API void        rfnn_session_free(rfnn_session* handle);

/* The grid the next inference fills. Clears every binding, since their shapes depended on it. */
RFNN_API rfnn_status rfnn_session_set_voxel_grid(rfnn_session* handle, uint32_t x, uint32_t y,
                                                 uint32_t z);

/* Register caller-owned memory for a tensor. Inputs hold METRIC values — the package's normalizer
 * is applied by the session, so a caller passes metres, radians and electronvolts and never learns
 * how the model was normalised. */
RFNN_API rfnn_status rfnn_session_bind_input(rfnn_session* handle, const char* name,
                                             rfnn_memory* memory);
RFNN_API rfnn_status rfnn_session_bind_output(rfnn_session* handle, const char* name,
                                              rfnn_memory* memory);

/* Run. Returns only once the device work is done, so a renderer needs no cross-API fence for the
 * write itself. A missing binding fails here and the message names EVERY one that is missing, not
 * the first. */
RFNN_API rfnn_status rfnn_session_infer(rfnn_session* handle);

/* The backend actually in use, as its name, or NULL for a NULL handle. */
RFNN_API const char* rfnn_session_backend(const rfnn_session* handle);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* RFNN_C_API_H */
