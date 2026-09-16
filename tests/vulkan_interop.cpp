// Vulkan → CUDA external memory, end to end.
//
// This test plays the part of the renderer. Unreal owns the Vulkan device and allocates the buffer
// its shader samples; here the test does exactly that — creates a device, allocates exportable
// memory, exports a file descriptor — and then hands it to the library the same way a plugin would.
//
// THE TEST LINKS THE VULKAN LOADER; THE LIBRARY DOES NOT. That asymmetry is the point of the design
// (plan D-2): importing is `cudaImportExternalMemory` on a descriptor the renderer already exported,
// so `rfnn` needs no Vulkan SDK and an engine plugin ships no loader on its behalf. The only role
// that genuinely needs Vulkan is the one this file is impersonating.
//
// Everything here skips cleanly when there is no Vulkan device, no CUDA device, or no device that
// both APIs can see — the suite has to pass on a machine without a GPU.
#include <RadFiled3D/nn/backends/cuda.hpp>
#include <RadFiled3D/nn/backends/onnx.hpp>
#include <RadFiled3D/nn/core/field_inference.hpp>
#include <RadFiled3D/nn/deploy.hpp>
#include <RadFiled3D/nn/memory/cuda.hpp>
#include <RadFiled3D/nn/memory/vk.hpp>

#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <optional>
#include <unistd.h>
#include <vector>

using namespace RadFiled3D::nn;

namespace {

/// The slice of Vulkan a renderer would already have. Torn down in reverse on destruction.
class VulkanHarness {
public:
    static std::optional<VulkanHarness> create(std::uint64_t bytes) {
        VulkanHarness h;
        if (!h.init(bytes)) return std::nullopt;
        return h;
    }

    VulkanHarness(VulkanHarness&& other) noexcept { *this = std::move(other); }
    VulkanHarness& operator=(VulkanHarness&& other) noexcept {
        std::swap(instance_, other.instance_);
        std::swap(device_, other.device_);
        std::swap(buffer_, other.buffer_);
        std::swap(memory_, other.memory_);
        std::swap(bytes_, other.bytes_);
        std::swap(uuid_, other.uuid_);
        return *this;
    }
    VulkanHarness(const VulkanHarness&) = delete;
    VulkanHarness& operator=(const VulkanHarness&) = delete;

    ~VulkanHarness() {
        if (device_ != VK_NULL_HANDLE) {
            if (buffer_ != VK_NULL_HANDLE) vkDestroyBuffer(device_, buffer_, nullptr);
            if (memory_ != VK_NULL_HANDLE) vkFreeMemory(device_, memory_, nullptr);
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
    }

    /// Export the allocation as an opaque FD. **The importer consumes it** — the harness does not
    /// close it, which is the contract stated on `memory::OpaqueFd`.
    std::optional<int> export_fd() const {
        auto get_fd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR"));
        if (get_fd == nullptr) return std::nullopt;
        VkMemoryGetFdInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        info.memory = memory_;
        info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        int fd = -1;
        if (get_fd(device_, &info, &fd) != VK_SUCCESS) return std::nullopt;
        return fd;
    }

    /// Read the allocation back through Vulkan — the half of the proof CUDA cannot fake.
    std::vector<float> read_back(std::size_t floats) const {
        void* mapped = nullptr;
        if (vkMapMemory(device_, memory_, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) return {};
        std::vector<float> out(floats);
        std::memcpy(out.data(), mapped, floats * sizeof(float));
        vkUnmapMemory(device_, memory_);
        return out;
    }

    const std::array<std::uint8_t, 16>& get_device_uuid() const noexcept { return uuid_; }
    std::uint64_t get_size_bytes() const noexcept { return bytes_; }

private:
    VulkanHarness() = default;

    bool init(std::uint64_t bytes) {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.apiVersion = VK_API_VERSION_1_1;  // external memory is core from 1.1
        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app;
        if (vkCreateInstance(&ici, nullptr, &instance_) != VK_SUCCESS) return false;

        std::uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance_, &count, nullptr);
        if (count == 0) return false;
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance_, &count, devices.data());

        // Pick the physical device CUDA can also see — matching by UUID is the whole point (D-4).
        VkPhysicalDevice chosen = VK_NULL_HANDLE;
        for (VkPhysicalDevice candidate : devices) {
            VkPhysicalDeviceIDProperties id{};
            id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
            VkPhysicalDeviceProperties2 props{};
            props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props.pNext = &id;
            vkGetPhysicalDeviceProperties2(candidate, &props);
            std::array<std::uint8_t, 16> uuid{};
            std::copy_n(id.deviceUUID, uuid.size(), uuid.begin());
            if (cuda::get_device_for_uuid(uuid) >= 0) {
                chosen = candidate;
                uuid_ = uuid;
                break;
            }
        }
        if (chosen == VK_NULL_HANDLE) return false;
        physical_ = chosen;

        float priority = 1.f;
        VkDeviceQueueCreateInfo queue{};
        queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue.queueFamilyIndex = 0;
        queue.queueCount = 1;
        queue.pQueuePriorities = &priority;
        const char* extensions[] = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &queue;
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = extensions;
        if (vkCreateDevice(chosen, &dci, nullptr, &device_) != VK_SUCCESS) return false;

        // A buffer the renderer would sample, declared shareable at creation.
        VkExternalMemoryBufferCreateInfo external{};
        external.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.pNext = &external;
        bci.size = bytes;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device_, &bci, nullptr, &buffer_) != VK_SUCCESS) return false;

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, buffer_, &requirements);
        // Host-visible, so the test can read back what CUDA wrote. A shipping renderer would use
        // device-local memory and a compute barrier instead; the import path is identical.
        const auto type = find_memory_type(requirements.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!type) return false;

        VkExportMemoryAllocateInfo export_info{};
        export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.pNext = &export_info;
        mai.allocationSize = requirements.size;
        mai.memoryTypeIndex = *type;
        if (vkAllocateMemory(device_, &mai, nullptr, &memory_) != VK_SUCCESS) return false;
        if (vkBindBufferMemory(device_, buffer_, memory_, 0) != VK_SUCCESS) return false;
        bytes_ = requirements.size;
        return true;
    }

    std::optional<std::uint32_t> find_memory_type(std::uint32_t mask, VkMemoryPropertyFlags want) const {
        VkPhysicalDeviceMemoryProperties props{};
        vkGetPhysicalDeviceMemoryProperties(physical_, &props);
        for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i)
            if ((mask & (1u << i)) != 0 && (props.memoryTypes[i].propertyFlags & want) == want) return i;
        return std::nullopt;
    }

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    std::uint64_t bytes_ = 0;
    std::array<std::uint8_t, 16> uuid_{};
};

constexpr std::size_t kFloats = 1024;
constexpr std::uint64_t kBytes = kFloats * sizeof(float);

}  // namespace

TEST(VulkanInterop, AVulkanDeviceAndACudaDeviceAgreeOnTheirUuid) {
    if (cuda::get_device_count() == 0) GTEST_SKIP() << "no CUDA device";
    const auto harness = VulkanHarness::create(kBytes);
    if (!harness) GTEST_SKIP() << "no Vulkan device that CUDA can also see";

    // The harness only succeeds by finding a Vulkan device whose UUID CUDA recognises, so reaching
    // here IS the claim: both APIs name the same GPU the same way.
    const int device = cuda::get_device_for_uuid(harness->get_device_uuid());
    ASSERT_GE(device, 0);
    EXPECT_EQ(cuda::get_device_uuid(device), harness->get_device_uuid());
    std::printf("  matched Vulkan device to CUDA ordinal %d (%s)\n", device,
                cuda::get_device_name(device).c_str());
}

/// The claim the whole design rests on: memory allocated by Vulkan, written by CUDA, read back
/// through Vulkan. If this passes, inference can write into a renderer's buffer.
TEST(VulkanInterop, CudaWritesIntoAVulkanAllocation) {
    if (cuda::get_device_count() == 0) GTEST_SKIP() << "no CUDA device";
    auto harness = VulkanHarness::create(kBytes);
    if (!harness) GTEST_SKIP() << "no Vulkan device that CUDA can also see";

    const auto fd = harness->export_fd();
    ASSERT_TRUE(fd.has_value()) << "vkGetMemoryFdKHR failed";

    memory::ExternalBuffer buffer;
    buffer.handle = memory::OpaqueFd{*fd};   // consumed by the import; the harness must not close it
    buffer.size_bytes = harness->get_size_bytes();
    buffer.offset_bytes = 0;
    buffer.region_bytes = kBytes;
    buffer.device_uuid = harness->get_device_uuid();

    memory::vk::ExternalMemory interop;
    ASSERT_TRUE(interop.supports(Backend::Cuda));
    const std::shared_ptr<memory::MemoryRef> imported = interop.import_buffer(buffer, Backend::Cuda);
    ASSERT_NE(imported, nullptr);

    // It comes back in the COMPUTE domain, because that is what a session can bind.
    EXPECT_EQ(imported->get_domain(), memory::Domain::Cuda);
    EXPECT_EQ(imported->get_size_bytes(), kBytes);
    EXPECT_EQ(imported->get_element_count(sizeof(float)), kFloats);

    const auto* as_cuda = dynamic_cast<const memory::cuda::MemoryRef*>(imported.get());
    ASSERT_NE(as_cuda, nullptr) << "the reference must BE a cuda::MemoryRef, not merely report its domain";
    ASSERT_NE(as_cuda->get_device_ptr(), 0u);

    // It KNOWS which card it landed on, which is what `Device::of(*imported)` reads and what lets a
    // session refuse memory from another one. A mapped external allocation is the case most likely to
    // answer differently from an ordinary `cudaMalloc`, so the recorded ordinal and what the driver
    // says about the same pointer are checked against each other here rather than assumed to agree.
    EXPECT_GE(imported->get_device_index(), 0) << "an imported buffer must know its device";
    EXPECT_EQ(memory::cuda::device_of(as_cuda->get_device_ptr()), imported->get_device_index());

    // Write a recognisable ramp from the CUDA side, into what Vulkan allocated.
    std::vector<float> expected(kFloats);
    for (std::size_t i = 0; i < kFloats; ++i) expected[i] = static_cast<float>(i) * 0.5f + 1.f;
    ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(as_cuda->get_device_ptr()), expected.data(), kBytes,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // ... and read it back through Vulkan. Nothing CUDA-side is consulted here.
    const std::vector<float> seen = harness->read_back(kFloats);
    ASSERT_EQ(seen.size(), kFloats);
    EXPECT_EQ(seen, expected) << "CUDA's write is not visible in the Vulkan allocation";
}

/// Releasing the last reference releases the import — the lifetime contract is a destructor, not a
/// rule someone has to remember (plan D-5).
TEST(VulkanInterop, ReleasingTheReferenceReleasesTheImport) {
    if (cuda::get_device_count() == 0) GTEST_SKIP() << "no CUDA device";
    auto harness = VulkanHarness::create(kBytes);
    if (!harness) GTEST_SKIP() << "no Vulkan device that CUDA can also see";

    // Import and drop, repeatedly. A leaked cudaExternalMemory_t or mapping would accumulate and
    // eventually fail; completing the loop is the evidence that neither does.
    for (int i = 0; i < 8; ++i) {
        const auto fd = harness->export_fd();
        ASSERT_TRUE(fd.has_value());
        memory::ExternalBuffer buffer;
        buffer.handle = memory::OpaqueFd{*fd};
        buffer.size_bytes = harness->get_size_bytes();
        buffer.region_bytes = kBytes;
        buffer.device_uuid = harness->get_device_uuid();

        memory::vk::ExternalMemory interop;
        auto imported = interop.import_buffer(buffer, Backend::Cuda);
        ASSERT_NE(imported, nullptr) << "import " << i << " failed";
    }
}

/// A device the allocation does not live on is refused, never silently copied (R-I1, plan D-4).
TEST(VulkanInterop, AnAllocationFromAnotherDeviceIsRefused) {
    if (cuda::get_device_count() == 0) GTEST_SKIP() << "no CUDA device";
    auto harness = VulkanHarness::create(kBytes);
    if (!harness) GTEST_SKIP() << "no Vulkan device that CUDA can also see";

    const auto fd = harness->export_fd();
    ASSERT_TRUE(fd.has_value());
    memory::ExternalBuffer buffer;
    buffer.handle = memory::OpaqueFd{*fd};
    buffer.size_bytes = harness->get_size_bytes();
    buffer.region_bytes = kBytes;
    buffer.device_uuid.fill(0xab);  // a GPU neither API has ever seen

    memory::vk::ExternalMemory interop;
    try {
        (void)interop.import_buffer(buffer, Backend::Cuda);
        FAIL() << "memory from an unknown device must not be imported";
    } catch (const Exception& err) {
        EXPECT_EQ(err.get_kind(), ErrorKind::InvalidArgument);
        EXPECT_NE(std::string(err.what()).find("UUID"), std::string::npos) << err.what();
    }
    ::close(*fd);  // a FAILED import does not consume the descriptor
}

// ── inference straight into the renderer's memory ────────────────────────────────────────────────

/// The goal of the whole design: a trained model runs on the GPU and its output lands in a buffer
/// Vulkan allocated, with no host round-trip. Inputs come from device memory too, so nothing in the
/// run crosses the bus.
///
/// Needs a real package. `RFNN_TEST_MODEL` overrides the path; without one the case skips, because a
/// checkout has no model in it.
TEST(VulkanInterop, InferenceWritesIntoTheRenderersBuffer) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";
    if (cuda::get_device_count() == 0) GTEST_SKIP() << "no CUDA device";

    const char* env = std::getenv("RFNN_TEST_MODEL");
    const std::string model_path =
        env ? env : "/mnt/data/models_nn/logs/pbrf-ds03-3779-noroi/pbrf-ds03-3779-noroi/models/PBRFNet.rf3m";
    if (!std::filesystem::exists(model_path)) GTEST_SKIP() << "no model at " << model_path;

    const deploy::Package package = deploy::Package::read_file(model_path);

    constexpr std::uint32_t kSide = 8;
    constexpr std::int64_t kQueries = kSide * kSide * kSide;

    // THE RENDERER'S ORDER, and the reason `Device::of` exists: the engine already decided which GPU
    // its buffer lives on, so the buffer is imported FIRST and the session is then built on whatever
    // device the import landed on. Choosing a device up front and hoping it matches is the
    // arrangement that produces a pointer valid only in the other card's context.
    auto harness = VulkanHarness::create(kQueries * sizeof(float));
    if (!harness) GTEST_SKIP() << "no Vulkan device that CUDA can also see";
    const auto fd = harness->export_fd();
    ASSERT_TRUE(fd.has_value());

    memory::ExternalBuffer external;
    external.handle = memory::OpaqueFd{*fd};
    external.size_bytes = harness->get_size_bytes();
    external.region_bytes = kQueries * sizeof(float);
    external.device_uuid = harness->get_device_uuid();

    memory::vk::ExternalMemory interop;
    // `flux` — the renderer's buffer, as an inference OUTPUT. This is the binding that matters.
    const std::shared_ptr<memory::MemoryRef> flux = interop.import_buffer(external, Backend::Cuda);

    auto session = load(package, Backend::Cuda, Device::of(*flux));
    EXPECT_EQ(session->get_device(), flux->get_device_index())
        << "the session must land on the card the renderer's memory is on";
    session->set_voxel_grid({kSide, kSide, kSide});

    // The inputs the trunk declares, in device memory, so the run touches no host buffer either.
    struct DeviceBuffer {
        void* ptr = nullptr;
        ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    };
    auto upload = [](const std::vector<float>& host, DeviceBuffer& owner) {
        const std::size_t bytes = host.size() * sizeof(float);
        if (cudaMalloc(&owner.ptr, bytes) != cudaSuccess) return std::shared_ptr<memory::MemoryRef>{};
        if (cudaMemcpy(owner.ptr, host.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess)
            return std::shared_ptr<memory::MemoryRef>{};
        return std::static_pointer_cast<memory::MemoryRef>(std::make_shared<memory::cuda::MemoryRef>(
            reinterpret_cast<std::uintptr_t>(owner.ptr), bytes));
    };
    auto device_input = [&upload](std::size_t floats, float fill, DeviceBuffer& owner) {
        return upload(std::vector<float>(floats, fill), owner);
    };

    DeviceBuffer position_mem, latent_mem, region_mem, spectrum_mem;
    // REAL voxel centres, not a constant. The package's field box is 1 m, so this is the same grid
    // a caller would fill through `FieldInference` — and it is what makes the flux vary across the
    // renderer's buffer instead of being one number repeated, which a broken position binding would
    // also produce.
    const auto position = upload(make_voxel_center_positions(FieldGeometry::cubic(kSide, 1.f)), position_mem);
    const auto latent = device_input(kQueries * 192, 0.01f, latent_mem);
    const auto region_state = device_input(14, 0.f, region_mem);
    const auto spectrum = device_input(kQueries * 32, 0.f, spectrum_mem);
    ASSERT_TRUE(position && latent && region_state && spectrum) << "cudaMalloc failed";

    session->bind_input("position", position);
    session->bind_input("latent", latent);
    session->bind_input("region_state", region_state);
    session->bind_output("flux", flux);
    session->bind_output("spectrum", spectrum);

    session->infer();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Read the prediction back THROUGH VULKAN. Nothing CUDA-side is consulted.
    const std::vector<float> seen = harness->read_back(kQueries);
    ASSERT_EQ(seen.size(), static_cast<std::size_t>(kQueries));
    EXPECT_TRUE(std::all_of(seen.begin(), seen.end(), [](float v) { return std::isfinite(v); }))
        << "the network wrote non-finite values into the renderer's buffer";
    EXPECT_NE(std::count_if(seen.begin(), seen.end(), [](float v) { return v != 0.f; }), 0)
        << "the buffer is still all zeros — nothing was written";
    // Each voxel was queried at its OWN centre, so the field must vary. A constant would mean the
    // position binding never reached the graph — which the previous constant-input form of this test
    // could not have detected.
    EXPECT_NE(*std::min_element(seen.begin(), seen.end()), *std::max_element(seen.begin(), seen.end()))
        << "every voxel came back identical — the positions did not reach the graph";
    std::printf("  flux[0..3] = %g %g %g %g  (min %g, max %g; read back through Vulkan)\n", seen[0], seen[1],
                seen[2], seen[3], *std::min_element(seen.begin(), seen.end()),
                *std::max_element(seen.begin(), seen.end()));
}

/// Every binding must be present before a run, and the diagnostic names them ALL at once (R-I1).
TEST(VulkanInterop, EveryMissingBindingIsReportedAtOnce) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";
    if (cuda::get_device_count() == 0) GTEST_SKIP() << "no CUDA device";
    const char* env = std::getenv("RFNN_TEST_MODEL");
    const std::string model_path =
        env ? env : "/mnt/data/models_nn/logs/pbrf-ds03-3779-noroi/pbrf-ds03-3779-noroi/models/PBRFNet.rf3m";
    if (!std::filesystem::exists(model_path)) GTEST_SKIP() << "no model at " << model_path;

    auto session = onnx::load(deploy::Package::read_file(model_path), Backend::Cuda, 0);
    session->set_voxel_grid({4, 4, 4});
    try {
        session->infer();
        FAIL() << "a run with nothing bound must not proceed";
    } catch (const Exception& err) {
        const std::string what = err.what();
        // All of them, not just the first.
        EXPECT_NE(what.find("position"), std::string::npos) << what;
        EXPECT_NE(what.find("latent"), std::string::npos) << what;
        EXPECT_NE(what.find("flux"), std::string::npos) << what;
        EXPECT_NE(what.find("spectrum"), std::string::npos) << what;
    }
}

/// The same run on TensorRT must land in the same buffer and agree with CUDA. TensorRT is reached
/// through ORT's execution provider, which dlopen's libnvinfer at run time — so this skips wherever
/// those runtime libraries are not on the loader path, which is a deployment question, not a build
/// one (plan D-1).
TEST(VulkanInterop, TensorRtWritesTheSameResultIntoTheRenderersBuffer) {
    if (!onnx::available()) GTEST_SKIP() << "built without ONNX Runtime";
    if (cuda::get_device_count() == 0) GTEST_SKIP() << "no CUDA device";
    const char* env = std::getenv("RFNN_TEST_MODEL");
    const std::string model_path =
        env ? env : "/mnt/data/models_nn/logs/pbrf-ds03-3779-noroi/pbrf-ds03-3779-noroi/models/PBRFNet.rf3m";
    if (!std::filesystem::exists(model_path)) GTEST_SKIP() << "no model at " << model_path;

    const deploy::Package package = deploy::Package::read_file(model_path);
    constexpr std::uint32_t kSide = 8;
    constexpr std::int64_t kQueries = kSide * kSide * kSide;

    auto run_on = [&](Backend backend) -> std::vector<float> {
        auto session = onnx::load(package, backend, 0);
        session->set_voxel_grid({kSide, kSide, kSide});
        auto harness = VulkanHarness::create(kQueries * sizeof(float));
        if (!harness) return {};
        const auto fd = harness->export_fd();
        if (!fd) return {};

        memory::ExternalBuffer external;
        external.handle = memory::OpaqueFd{*fd};
        external.size_bytes = harness->get_size_bytes();
        external.region_bytes = kQueries * sizeof(float);
        external.device_uuid = harness->get_device_uuid();
        memory::vk::ExternalMemory interop;
        const auto flux = interop.import_buffer(external, Backend::Cuda);

        std::vector<void*> owned;
        auto device_input = [&owned](std::size_t floats, float fill) {
            std::vector<float> host(floats, fill);
            void* ptr = nullptr;
            cudaMalloc(&ptr, floats * sizeof(float));
            cudaMemcpy(ptr, host.data(), floats * sizeof(float), cudaMemcpyHostToDevice);
            owned.push_back(ptr);
            return std::static_pointer_cast<memory::MemoryRef>(std::make_shared<memory::cuda::MemoryRef>(
                reinterpret_cast<std::uintptr_t>(ptr), floats * sizeof(float)));
        };
        session->bind_input("position", device_input(kQueries * 3, 0.5f));
        session->bind_input("latent", device_input(kQueries * 192, 0.01f));
        session->bind_input("region_state", device_input(14, 0.f));
        session->bind_output("flux", flux);
        session->bind_output("spectrum", device_input(kQueries * 32, 0.f));
        session->infer();
        cudaDeviceSynchronize();
        auto out = harness->read_back(kQueries);
        for (void* p : owned) cudaFree(p);
        return out;
    };

    const std::vector<float> from_cuda = run_on(Backend::Cuda);
    if (from_cuda.empty()) GTEST_SKIP() << "no Vulkan device that CUDA can also see";

    std::vector<float> from_trt;
    try {
        from_trt = run_on(Backend::TensorRt);
    } catch (const std::exception& err) {
        GTEST_SKIP() << "TensorRT provider unavailable at run time: " << err.what();
    }
    ASSERT_EQ(from_trt.size(), from_cuda.size());
    for (std::size_t i = 0; i < from_cuda.size(); ++i)
        // Not bit-exact: TensorRT picks its own kernels and fuses differently. Agreement to a
        // relative 1e-3 is what "the same model" means across two providers.
        EXPECT_NEAR(from_trt[i], from_cuda[i], std::abs(from_cuda[i]) * 1e-3f + 1e-9f) << "at " << i;
    std::printf("  TensorRT flux[0] = %g, CUDA flux[0] = %g\n", from_trt[0], from_cuda[0]);
}
