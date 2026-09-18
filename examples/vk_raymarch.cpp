// Ray-march a predicted radiation field out of the renderer's OWN memory.
//
// What this demonstrates, end to end and on one allocation:
//
//   1. Vulkan allocates the voxel volume and exports a handle for it, the way an engine would.
//   2. The handle is described as an ordinary `memory::MemoryRef`, and the CUDA session ADOPTS it —
//      importing once, giving a second view of the same bytes.
//   3. A real `.rf3m` model is loaded and run for a random beam configuration, writing `flux`
//      straight into that Vulkan allocation. No host round trip, no copy.
//   4. A Vulkan compute shader ray-marches the very same buffer and writes an image.
//
// The point is step 4 reading what step 3 wrote WITHOUT anything moving in between. If the import
// were a copy, the picture would be of stale memory.
//
// Run: vk_raymarch [--model PATH] [--voxels N] [--width W] [--height H] [--seed S] [--out FILE]
#include <RadFiled3D/nn.hpp>

#include <cuda_runtime_api.h>

#include <vulkan/vulkan.h>
#ifndef RFNN_NO_PRESENT
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <vulkan/vulkan_xlib.h>
#endif

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "raymarch_spv.h"

using namespace RadFiled3D::nn;

namespace {

/// The trunk's latent width and region-state size for a PBRFNet-shaped package. Read from the
/// graphs rather than guessed would be better; these packages declare neither, which is the same
/// interface gap `encode_beam` explains.
constexpr std::uint32_t kLatentWidth = 192;
constexpr std::uint32_t kRegionState = 14;
/// The encoder's tube spectrum is finer than the one the package declares.
constexpr std::size_t kEncoderSpectrumBins = 150;

struct Options {
    std::string model = "/mnt/data/models_nn/logs/pbrf-ds03-3779-noroi/pbrf-ds03-3779-noroi/models/PBRFNet.rf3m";
    std::uint32_t voxels = 48;
    std::uint32_t width = 900;
    std::uint32_t height = 600;
    unsigned seed = 0;                 // 0 = random
    std::string out;          // empty: do not write a file
    bool present = true;      // a window, unless --offscreen
    float region_width = 0.25f;   // the trunk's positional-encoding scale
    std::uint32_t frames = 0;     // >0: offscreen, sweep the tube over a full circle
    float tilt = 0.25f;           // elevation of the sweep, radians above the horizon
};

[[noreturn]] void die(const std::string& what) {
    std::cerr << "vk_raymarch: " << what << "\n";
    std::exit(2);
}

#define VK_OK(expr, what) do { if ((expr) != VK_SUCCESS) die(std::string("Vulkan: ") + (what)); } while (0)

/// Everything a renderer would already own: a device CUDA can also see, an exportable volume
/// buffer, and a compute pipeline to march it.
class Renderer {
public:
    Renderer(std::uint32_t voxels, std::uint32_t width, std::uint32_t height, bool present)
        : voxels_(voxels), width_(width), height_(height), present_(present) {
        if (present_) open_window();
        create_device();
        if (present_) create_swapchain();
        volume_bytes_ = std::uint64_t(voxels) * voxels * voxels * sizeof(float);
        volume_ = create_buffer(volume_bytes_, /*exportable=*/true, volume_memory_, volume_allocated_);
        image_bytes_ = std::uint64_t(width) * height * sizeof(std::uint32_t);
        image_ = create_buffer(image_bytes_, /*exportable=*/false, image_memory_, image_allocated_);
        create_pipeline();
    }

    ~Renderer() {
        if (device_ == VK_NULL_HANDLE) return;
        vkDeviceWaitIdle(device_);
        if (acquired_ != VK_NULL_HANDLE) vkDestroySemaphore(device_, acquired_, nullptr);
        if (rendered_ != VK_NULL_HANDLE) vkDestroySemaphore(device_, rendered_, nullptr);
        if (swapchain_ != VK_NULL_HANDLE) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        if (pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, pool_, nullptr);
        if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline_, nullptr);
        if (layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, layout_, nullptr);
        if (set_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
        if (descriptors_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, descriptors_, nullptr);
        if (module_ != VK_NULL_HANDLE) vkDestroyShaderModule(device_, module_, nullptr);
        for (VkBuffer b : {volume_, image_}) if (b != VK_NULL_HANDLE) vkDestroyBuffer(device_, b, nullptr);
        for (VkDeviceMemory m : {volume_memory_, image_memory_}) if (m != VK_NULL_HANDLE) vkFreeMemory(device_, m, nullptr);
        vkDestroyDevice(device_, nullptr);
        if (surface_ != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);
#ifndef RFNN_NO_PRESENT
        if (display_ != nullptr) XCloseDisplay(display_);
#endif
    }

    /// Export the volume as an opaque descriptor. **The importer consumes it** — that is the
    /// contract on `memory::OpaqueFd`, so this side must not close it.
    int export_volume_fd() const {
        auto get_fd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR"));
        if (get_fd == nullptr) die("vkGetMemoryFdKHR unavailable");
        VkMemoryGetFdInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        info.memory = volume_memory_;
        info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        int fd = -1;
        VK_OK(get_fd(device_, &info, &fd), "vkGetMemoryFdKHR");
        return fd;
    }

    std::uint64_t get_volume_allocated() const noexcept { return volume_allocated_; }
    std::uint64_t get_volume_bytes() const noexcept { return volume_bytes_; }
    const std::array<std::uint8_t, 16>& get_device_uuid() const noexcept { return uuid_; }

    /// Wipe the shared allocation through Vulkan. If inference then refills it, the write is real.
    void zero_volume() {
        void* mapped = nullptr;
        VK_OK(vkMapMemory(device_, volume_memory_, 0, VK_WHOLE_SIZE, 0, &mapped), "map volume");
        std::memset(mapped, 0, volume_bytes_);
        vkUnmapMemory(device_, volume_memory_);
    }

    /// What Vulkan itself sees in the shared allocation — the half of the proof CUDA cannot fake.
    std::vector<float> read_volume() const {
        void* mapped = nullptr;
        VK_OK(vkMapMemory(device_, volume_memory_, 0, VK_WHOLE_SIZE, 0, &mapped), "map volume");
        std::vector<float> out(volume_bytes_ / sizeof(float));
        std::memcpy(out.data(), mapped, volume_bytes_);
        vkUnmapMemory(device_, volume_memory_);
        return out;
    }

    struct View { float yaw, pitch, distance, lo, hi; };

    VkCommandBuffer begin_commands() {
        VkCommandBufferAllocateInfo cbi{};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbi.commandPool = pool_;
        cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbi.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VK_OK(vkAllocateCommandBuffers(device_, &cbi, &cmd), "allocate command buffer");
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_OK(vkBeginCommandBuffer(cmd, &begin), "begin command buffer");
        return cmd;
    }

    /// The dispatch itself, shared by the offscreen and the on-screen paths so both show exactly
    /// the same image.
    void record_raymarch(VkCommandBuffer cmd, const View& view) {
        struct Push {
            std::uint32_t vx, vy, vz, width, height;
            float yaw, pitch, distance, lo, hi;
        } push{voxels_, voxels_, voxels_, width_, height_, view.yaw, view.pitch,
               view.distance, view.lo, view.hi};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &set_, 0, nullptr);
        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, (width_ + 7) / 8, (height_ + 7) / 8, 1);
    }

    /// Dispatch the ray-march over the shared volume and return the rendered pixels.
    std::vector<std::uint32_t> render(const View& view) {
        VkCommandBuffer cmd = begin_commands();
        record_raymarch(cmd, view);
        VK_OK(vkEndCommandBuffer(cmd), "end command buffer");

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        VK_OK(vkQueueSubmit(queue_, 1, &submit, VK_NULL_HANDLE), "submit");
        VK_OK(vkQueueWaitIdle(queue_), "wait idle");
        vkFreeCommandBuffers(device_, pool_, 1, &cmd);

        void* mapped = nullptr;
        VK_OK(vkMapMemory(device_, image_memory_, 0, VK_WHOLE_SIZE, 0, &mapped), "map image");
        std::vector<std::uint32_t> pixels(std::size_t(width_) * height_);
        std::memcpy(pixels.data(), mapped, image_bytes_);
        vkUnmapMemory(device_, image_memory_);
        return pixels;
    }


private:
    void create_device() {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "rfnn vk_raymarch";
        app.apiVersion = VK_API_VERSION_1_1;   // external memory is core from 1.1
        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app;
        const char* surface_extensions[] = {VK_KHR_SURFACE_EXTENSION_NAME,
                                            VK_KHR_XLIB_SURFACE_EXTENSION_NAME};
        if (present_) {
            ici.enabledExtensionCount = 2;
            ici.ppEnabledExtensionNames = surface_extensions;
        }
        VK_OK(vkCreateInstance(&ici, nullptr, &instance_), "vkCreateInstance");
#ifndef RFNN_NO_PRESENT
        if (present_) {
            VkXlibSurfaceCreateInfoKHR sci{};
            sci.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
            sci.dpy = display_;
            sci.window = window_;
            VK_OK(vkCreateXlibSurfaceKHR(instance_, &sci, nullptr, &surface_), "xlib surface");
        }
#endif

        std::uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance_, &count, nullptr);
        if (count == 0) die("no Vulkan physical device");
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance_, &count, devices.data());

        // The device CUDA can ALSO see, matched by UUID. Sharing memory with a card the compute
        // backend cannot address is the one mistake this check exists to prevent.
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
                physical_ = candidate;
                uuid_ = uuid;
                name_ = props.properties.deviceName;
                break;
            }
        }
        if (physical_ == VK_NULL_HANDLE) die("no Vulkan device that CUDA can also see");

        float priority = 1.f;
        VkDeviceQueueCreateInfo queue{};
        queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue.queueFamilyIndex = 0;
        queue.queueCount = 1;
        queue.pQueuePriorities = &priority;
        const char* extensions[] = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                                    VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &queue;
        dci.enabledExtensionCount = present_ ? 2u : 1u;
        dci.ppEnabledExtensionNames = extensions;
        VK_OK(vkCreateDevice(physical_, &dci, nullptr, &device_), "vkCreateDevice");
        vkGetDeviceQueue(device_, 0, 0, &queue_);
    }

    std::uint32_t find_memory_type(std::uint32_t mask, VkMemoryPropertyFlags want) const {
        VkPhysicalDeviceMemoryProperties props{};
        vkGetPhysicalDeviceMemoryProperties(physical_, &props);
        for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i)
            if ((mask & (1u << i)) != 0 && (props.memoryTypes[i].propertyFlags & want) == want) return i;
        die("no suitable Vulkan memory type");
    }

    VkBuffer create_buffer(std::uint64_t bytes, bool exportable, VkDeviceMemory& memory,
                           std::uint64_t& allocated) {
        VkExternalMemoryBufferCreateInfo external{};
        external.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.pNext = exportable ? &external : nullptr;
        bci.size = bytes;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer buffer = VK_NULL_HANDLE;
        VK_OK(vkCreateBuffer(device_, &bci, nullptr, &buffer), "vkCreateBuffer");

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, buffer, &requirements);
        // Host-visible so the example can show what Vulkan sees. A shipping renderer would use
        // device-local memory and a barrier; the import path is identical either way.
        VkExportMemoryAllocateInfo export_info{};
        export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.pNext = exportable ? &export_info : nullptr;
        mai.allocationSize = requirements.size;
        mai.memoryTypeIndex = find_memory_type(
            requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_OK(vkAllocateMemory(device_, &mai, nullptr, &memory), "vkAllocateMemory");
        VK_OK(vkBindBufferMemory(device_, buffer, memory, 0), "vkBindBufferMemory");
        allocated = requirements.size;
        return buffer;
    }

#ifndef RFNN_NO_PRESENT
    /// An X window, because the point of this example is to be LOOKED at. Xlib rather than a
    /// toolkit: the headers are already here and a window plus mouse events is all that is needed,
    /// so a dependency would buy nothing.
    void open_window() {
        display_ = XOpenDisplay(nullptr);
        if (display_ == nullptr) die("cannot open the X display (is DISPLAY set?)");
        const int screen = DefaultScreen(display_);
        window_ = XCreateSimpleWindow(display_, RootWindow(display_, screen), 0, 0, width_, height_, 0,
                                      BlackPixel(display_, screen), BlackPixel(display_, screen));
        XStoreName(display_, window_, "radfiled3d-nn — predicted field, ray-marched from shared memory");
        XSelectInput(display_, window_,
                     ExposureMask | KeyPressMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask |
                         StructureNotifyMask);
        close_atom_ = XInternAtom(display_, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(display_, window_, &close_atom_, 1);
        XMapWindow(display_, window_);
        XFlush(display_);
    }

    void create_swapchain() {
        VkBool32 supported = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(physical_, 0, surface_, &supported);
        if (supported != VK_TRUE) die("queue family 0 cannot present to this surface");

        VkSurfaceCapabilitiesKHR caps{};
        VK_OK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_, surface_, &caps), "surface caps");
        extent_ = caps.currentExtent.width != 0xFFFFFFFFu ? caps.currentExtent
                                                          : VkExtent2D{width_, height_};

        std::uint32_t formats = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &formats, nullptr);
        std::vector<VkSurfaceFormatKHR> available(formats);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &formats, available.data());
        format_ = available.front();
        for (const auto& f : available)
            if (f.format == VK_FORMAT_B8G8R8A8_UNORM) { format_ = f; break; }

        VkSwapchainCreateInfoKHR sci{};
        sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        sci.surface = surface_;
        sci.minImageCount = std::max(caps.minImageCount, 2u);
        sci.imageFormat = format_.format;
        sci.imageColorSpace = format_.colorSpace;
        sci.imageExtent = extent_;
        sci.imageArrayLayers = 1;
        // The compute shader writes a BUFFER, which is then copied into the acquired image. Copying
        // rather than writing the swapchain image directly keeps the shader identical to the
        // offscreen path and asks nothing of the driver beyond TRANSFER_DST.
        sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        sci.preTransform = caps.currentTransform;
        sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;   // always supported, and vsync suits an explorer
        sci.clipped = VK_TRUE;
        VK_OK(vkCreateSwapchainKHR(device_, &sci, nullptr, &swapchain_), "vkCreateSwapchainKHR");

        std::uint32_t count = 0;
        vkGetSwapchainImagesKHR(device_, swapchain_, &count, nullptr);
        swapchain_images_.resize(count);
        vkGetSwapchainImagesKHR(device_, swapchain_, &count, swapchain_images_.data());

        VkSemaphoreCreateInfo sem{};
        sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_OK(vkCreateSemaphore(device_, &sem, nullptr, &acquired_), "semaphore");
        VK_OK(vkCreateSemaphore(device_, &sem, nullptr, &rendered_), "semaphore");
    }

    void barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to,
                 VkAccessFlags src, VkAccessFlags dst, VkPipelineStageFlags src_stage,
                 VkPipelineStageFlags dst_stage) {
        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = from;
        b.newLayout = to;
        b.srcAccessMask = src;
        b.dstAccessMask = dst;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
    }

public:
    /// What the user did since the last frame.
    struct Input {
        bool quit = false;
        bool reroll = false;    ///< ask for a new beam configuration
        bool reset = false;
        bool toggle_orbit = false;
    };

    /// Drain the X queue and fold it into the camera. Returns false once the window should close.
    bool poll(View& view, Input& input) {
        input = Input{};
        while (XPending(display_) > 0) {
            XEvent e;
            XNextEvent(display_, &e);
            switch (e.type) {
                case ClientMessage:
                    if (Atom(e.xclient.data.l[0]) == close_atom_) input.quit = true;
                    break;
                case KeyPress: {
                    const KeySym key = XLookupKeysym(&e.xkey, 0);
                    if (key == XK_Escape || key == XK_q) input.quit = true;
                    else if (key == XK_n || key == XK_space) input.reroll = true;
                    else if (key == XK_r) input.reset = true;
                    else if (key == XK_o) input.toggle_orbit = true;
                    break;
                }
                case ButtonPress:
                    if (e.xbutton.button == Button1) { dragging_ = true; last_x_ = e.xbutton.x; last_y_ = e.xbutton.y; }
                    // Wheel up / down.
                    else if (e.xbutton.button == Button4) view.distance = std::max(0.6f, view.distance * 0.9f);
                    else if (e.xbutton.button == Button5) view.distance = std::min(8.0f, view.distance * 1.1f);
                    break;
                case ButtonRelease:
                    if (e.xbutton.button == Button1) dragging_ = false;
                    break;
                case MotionNotify:
                    if (dragging_) {
                        view.yaw += float(e.xmotion.x - last_x_) * 0.008f;
                        view.pitch = std::clamp(view.pitch + float(e.xmotion.y - last_y_) * 0.008f,
                                                -1.5f, 1.5f);
                        last_x_ = e.xmotion.x;
                        last_y_ = e.xmotion.y;
                    }
                    break;
                default: break;
            }
        }
        return !input.quit;
    }

    /// True while the left button is held, so an automatic orbit can stand aside.
    bool is_dragging() const noexcept { return dragging_; }

    /// Ray-march the shared volume and show it.
    void present(const View& view) {
        std::uint32_t index = 0;
        const VkResult acquired =
            vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, acquired_, VK_NULL_HANDLE, &index);
        if (acquired == VK_ERROR_OUT_OF_DATE_KHR) return;   // the window is being resized; skip a frame
        if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) die("vkAcquireNextImageKHR");

        VkCommandBuffer cmd = begin_commands();
        record_raymarch(cmd, view);
        // The compute writes into a buffer; the copy must wait for it.
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             1, &mb, 0, nullptr, 0, nullptr);

        barrier(cmd, swapchain_images_[index], VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {extent_.width, extent_.height, 1};
        vkCmdCopyBufferToImage(cmd, image_, swapchain_images_[index],
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        barrier(cmd, swapchain_images_[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        VK_OK(vkEndCommandBuffer(cmd), "end command buffer");

        const VkPipelineStageFlags wait = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &acquired_;
        submit.pWaitDstStageMask = &wait;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &rendered_;
        VK_OK(vkQueueSubmit(queue_, 1, &submit, VK_NULL_HANDLE), "submit");

        VkPresentInfoKHR present{};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &rendered_;
        present.swapchainCount = 1;
        present.pSwapchains = &swapchain_;
        present.pImageIndices = &index;
        vkQueuePresentKHR(queue_, &present);
        VK_OK(vkQueueWaitIdle(queue_), "wait idle");
        vkFreeCommandBuffers(device_, pool_, 1, &cmd);
    }

private:
#else
    void open_window() { die("built without presentation support"); }
    void create_swapchain() {}
#endif

    void create_pipeline() {
        VkShaderModuleCreateInfo smi{};
        smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smi.codeSize = sizeof(kRaymarchSpv);
        smi.pCode = kRaymarchSpv;
        VK_OK(vkCreateShaderModule(device_, &smi, nullptr, &module_), "vkCreateShaderModule");

        VkDescriptorSetLayoutBinding bindings[2]{};
        for (std::uint32_t i = 0; i < 2; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dsl{};
        dsl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsl.bindingCount = 2;
        dsl.pBindings = bindings;
        VK_OK(vkCreateDescriptorSetLayout(device_, &dsl, nullptr, &set_layout_), "descriptor set layout");

        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        range.size = 10 * sizeof(std::uint32_t);
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &set_layout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &range;
        VK_OK(vkCreatePipelineLayout(device_, &pli, nullptr, &layout_), "pipeline layout");

        VkComputePipelineCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = module_;
        cpi.stage.pName = "main";
        cpi.layout = layout_;
        VK_OK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline_), "compute pipeline");

        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
        VkDescriptorPoolCreateInfo dpi{};
        dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &size;
        VK_OK(vkCreateDescriptorPool(device_, &dpi, nullptr, &descriptors_), "descriptor pool");

        VkDescriptorSetAllocateInfo dsa{};
        dsa.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsa.descriptorPool = descriptors_;
        dsa.descriptorSetCount = 1;
        dsa.pSetLayouts = &set_layout_;
        VK_OK(vkAllocateDescriptorSets(device_, &dsa, &set_), "descriptor set");

        VkDescriptorBufferInfo infos[2]{{volume_, 0, volume_bytes_}, {image_, 0, image_bytes_}};
        VkWriteDescriptorSet writes[2]{};
        for (std::uint32_t i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set_;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);

        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = 0;
        VK_OK(vkCreateCommandPool(device_, &cpci, nullptr, &pool_), "command pool");
    }

public:
    std::string name_;

private:
    std::uint32_t voxels_, width_, height_;
    bool present_ = false;
#ifndef RFNN_NO_PRESENT
    Display* display_ = nullptr;
    Window window_{};
    Atom close_atom_{};
    bool dragging_ = false;
    int last_x_ = 0, last_y_ = 0;
#endif
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkSurfaceFormatKHR format_{};
    VkExtent2D extent_{};
    std::vector<VkImage> swapchain_images_;
    VkSemaphore acquired_ = VK_NULL_HANDLE, rendered_ = VK_NULL_HANDLE;
    std::uint64_t volume_bytes_ = 0, volume_allocated_ = 0, image_bytes_ = 0, image_allocated_ = 0;
    std::array<std::uint8_t, 16> uuid_{};
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkBuffer volume_ = VK_NULL_HANDLE, image_ = VK_NULL_HANDLE;
    VkDeviceMemory volume_memory_ = VK_NULL_HANDLE, image_memory_ = VK_NULL_HANDLE;
    VkShaderModule module_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptors_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
};

void write_ppm(const std::string& path, const std::vector<std::uint32_t>& pixels,
               std::uint32_t width, std::uint32_t height) {
    std::ofstream out(path, std::ios::binary);
    if (!out) die("cannot write " + path);
    out << "P6\n" << width << " " << height << "\n255\n";
    for (std::uint32_t p : pixels) {
        const char rgb[3]{char(p & 0xFF), char((p >> 8) & 0xFF), char((p >> 16) & 0xFF)};
        out.write(rgb, 3);
    }
}

/// One random beam configuration, as a user moving the C-arm would produce.
struct Beam {
    std::vector<float> direction{std::vector<float>(3)};
    std::vector<float> distance{std::vector<float>(1)};
    std::vector<float> spectrum{std::vector<float>(kEncoderSpectrumBins)};
};

Beam roll_beam(std::mt19937& rng) {
    std::uniform_real_distribution<float> angle(-std::numbers::pi_v<float>, std::numbers::pi_v<float>);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    Beam beam;
    const float theta = angle(rng), phi = std::acos(2.f * unit(rng) - 1.f);
    beam.direction = {std::sin(phi) * std::cos(theta), std::sin(phi) * std::sin(theta), std::cos(phi)};
    beam.distance = {0.6f + 0.6f * unit(rng)};
    // A plausible tube spectrum: a broad bremsstrahlung-like hump, normalised.
    float total = 0.f;
    for (std::size_t i = 0; i < beam.spectrum.size(); ++i) {
        const float x = float(i) / float(beam.spectrum.size() - 1);
        beam.spectrum[i] = std::max(0.f, x * std::exp(-3.f * x)) + 0.01f * unit(rng);
        total += beam.spectrum[i];
    }
    for (float& v : beam.spectrum) v /= total;
    return beam;
}

/// Window the colour ramp on what is actually in the volume.
///
/// A percentile floor rather than the minimum, so a handful of near-zero voxels cannot stretch the
/// ramp flat. Recomputed for every beam, because the range moves with the configuration.
void window_for(const std::vector<float>& volume, Renderer::View& view) {
    std::vector<float> sorted;
    sorted.reserve(volume.size());
    for (float v : volume) if (v > 0.f) sorted.push_back(v);
    if (sorted.empty()) die("the shared allocation is empty — inference did not reach the renderer");
    std::sort(sorted.begin(), sorted.end());
    view.lo = std::log(sorted[sorted.size() / 100] + 1e-30f);
    view.hi = std::log(sorted[(sorted.size() * 999) / 1000] + 1e-30f);
}

/// Run the package's `encoding_config` graph to obtain the trunk's `region_state`.
///
/// NOT optional, and not zeros. `region_state` configures the trunk's positional encoding over the
/// region of interest, and feeding zeros puts the network far outside anything it was trained on:
/// the field comes back almost constant (a 1.7x spread over the whole volume) with its peak in a
/// corner. Driven properly the same model produces ~5800x and puts the peak where the beam enters.
/// A composition would wire this automatically; this package predates one, so it is run here.
std::vector<float> encode_region(const deploy::Package& package, float region_width) {
    const auto graph = package.get_graph("encoding_config");
    if (!graph) die("this package carries no `encoding_config` graph");

    deploy::PackageBuilder config;
    config.provenance(package.provenance.dataset, "vk_raymarch", package.provenance.physics)
        .field_dimensions_m(package.geometry.field_dimensions_m)
        .input("region_width", deploy::Semantic::from_name("region_width"), {1}).done()
        .output("region_state", deploy::Semantic::from_name("region_state"), {kRegionState}).done()
        .graph("trunk", deploy::bytes(graph->begin(), graph->end()));

    auto session = load(config.build(), Backend::Cpu, Device::automatic());
    session->set_voxel_grid({1, 1, 1});
    std::vector<float> width{region_width};
    session->bind_input("region_width", memory::host::MemoryRef::of(std::span<float>(width)));
    std::vector<float> state(kRegionState);
    session->bind_output("region_state", memory::host::MemoryRef::of(std::span<float>(state)));
    session->infer();
    return state;
}

/// The package's own `beam_encoder` graph, kept LOADED.
///
/// WHY A CLASS. Sweeping the tube means re-encoding every frame, and building a package and an ORT
/// session per call costs hundreds of milliseconds — the animation would be a slideshow. The
/// session is built once and its inputs are bound once to buffers that never move, so a frame is
/// "mutate the direction, infer" and nothing else.
///
/// WHY IT EXISTS AT ALL. These packages predate `deploy::Composition`: `beam_encoder` and `trunk`
/// ship as separate graphs with no wiring block, so the runtime cannot connect them and the
/// declared interface (`beam_direction`, `tube_spectrum`, ...) does not match the trunk's own
/// inputs (`latent`, `region_state`). That gap is a known, tested property — see
/// `FieldInference.TheV1InterfaceGapIsReportedAndNotSilent`. Given a composition, binding
/// `beam_direction` would reach the encoder through the wiring and this class would not exist.
class BeamEncoder {
public:
    BeamEncoder(const deploy::Package& package, std::size_t spectrum_bins, std::uint32_t latent_width)
        : spectrum_(spectrum_bins), latent_(latent_width) {
        const auto graph = package.get_graph("beam_encoder");
        if (!graph) die("this package carries no `beam_encoder` graph");

        deploy::PackageBuilder encoder;
        encoder.provenance(package.provenance.dataset, "vk_raymarch", package.provenance.physics)
            .field_dimensions_m(package.geometry.field_dimensions_m)
            // The GRAPH's names, because that is what an ONNX session binds by.
            .input("direction", deploy::Semantic::BeamDirection, {3}).done()
            .input("distance", deploy::Semantic::SourceDistance, {1}).done()
            .input("spectrum", deploy::Semantic::from_name("tube_spectrum"),
                   {std::uint32_t(spectrum_bins)}).done()
            .output("linear_5", deploy::Semantic::from_name("latent"), {latent_width}).done()
            .graph("trunk", deploy::bytes(graph->begin(), graph->end()));

        session_ = load(encoder.build(), Backend::Cpu, Device::automatic());
        session_->set_voxel_grid({1, 1, 1});
        // Bound ONCE, to buffers this object owns. A frame changes what is IN them, never where
        // they are — the discipline the runtime already applies to its own stage buffers.
        session_->bind_input("direction", memory::host::MemoryRef::of(std::span<float>(direction_)));
        session_->bind_input("distance", memory::host::MemoryRef::of(std::span<float>(distance_)));
        session_->bind_input("spectrum", memory::host::MemoryRef::of(std::span<float>(spectrum_)));
        session_->bind_output("linear_5", memory::host::MemoryRef::of(std::span<float>(latent_)));
    }

    void set_spectrum(const std::vector<float>& spectrum) { spectrum_ = spectrum; }
    void set_distance(float metres) { distance_[0] = metres; }

    const std::vector<float>& encode(const std::array<float, 3>& direction) {
        std::copy(direction.begin(), direction.end(), direction_.begin());
        session_->infer();
        return latent_;
    }

private:
    std::vector<float> direction_{0.f, 0.f, 1.f};
    std::vector<float> distance_{1.f};
    std::vector<float> spectrum_;
    std::vector<float> latent_;
    std::shared_ptr<InferenceSession> session_;
};

/// A tube position on a circle around the field, the way a C-arm sweeps.
///
/// `tilt` lifts the orbit out of the horizontal plane, so the sweep is not a degenerate great
/// circle through the cube's equator.
std::array<float, 3> tube_direction(float angle, float tilt) {
    const float c = std::cos(tilt);
    std::array<float, 3> d{c * std::cos(angle), std::sin(tilt), c * std::sin(angle)};
    const float length = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    for (float& v : d) v /= length;
    return d;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--model") o.model = next();
        else if (a == "--voxels") o.voxels = std::stoul(next());
        else if (a == "--width") o.width = std::stoul(next());
        else if (a == "--height") o.height = std::stoul(next());
        else if (a == "--seed") o.seed = std::stoul(next());
        else if (a == "--region") o.region_width = std::stof(next());
        else if (a == "--frames") { o.frames = std::stoul(next()); o.present = false; }
        else if (a == "--tilt") o.tilt = std::stof(next());
        else if (a == "--out") o.out = next();
        else if (a == "--offscreen") { o.present = false; if (o.out.empty()) o.out = "raymarch.ppm"; }
        else die("unknown argument " + a);
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) try {
    const Options options = parse(argc, argv);
    if (!std::filesystem::exists(options.model)) die("no model at " + options.model);
    if (cuda::get_device_count() == 0) die("no CUDA device");

    // ── 1. the renderer's volume, allocated by Vulkan and exportable ────────────────────────
    Renderer renderer(options.voxels, options.width, options.height, options.present);
    std::cout << "device      " << renderer.name_ << "\n";
    const std::uint32_t side = options.voxels;
    const std::int64_t queries = std::int64_t(side) * side * side;

    // ── 2. describe it as a MemoryRef; the backend adopts it ────────────────────────────────
    memory::ExternalOrigin origin;
    origin.handle = memory::OpaqueFd{renderer.export_volume_fd()};
    origin.size_bytes = renderer.get_volume_allocated();
    origin.region_bytes = renderer.get_volume_bytes();
    origin.device_uuid = renderer.get_device_uuid();
    auto exported = std::make_shared<memory::ExportedMemoryRef>(memory::Domain::Vulkan, origin);

    const ComputeBackend& cuda_backend = get_compute_backend(Backend::Cuda);
    const std::shared_ptr<memory::MemoryRef> flux = cuda_backend.adopt(exported, -1);
    std::cout << "adopted     Vulkan allocation -> " << memory::to_string(flux->get_domain())
              << " device " << flux->get_device_index() << "\n";

    // Asking again must not map it twice: the registry hands back the reference that exists.
    if (cuda_backend.adopt(exported, flux->get_device_index()).get() != flux.get())
        die("adopting the same allocation twice produced two mappings");

    // ── 3. the model, on the card the renderer's memory is already on ───────────────────────
    const deploy::Package package = deploy::Package::read_file(options.model);
    auto session = load(package, Backend::Cuda, Device::of(*flux));
    session->set_voxel_grid({side, side, side});
    std::cout << "model       " << std::filesystem::path(options.model).filename().string()
              << "  (" << package.provenance.dataset << ")\n";

    // A random beam configuration, as a user moving the C-arm would produce.
    std::random_device entropy;
    const unsigned seed = options.seed != 0 ? options.seed : entropy();
    std::mt19937 rng(seed);
    const Beam beam = roll_beam(rng);
    const std::vector<float>& direction = beam.direction;
    const std::vector<float>& distance = beam.distance;
    const std::vector<float>& spectrum = beam.spectrum;

    std::cout << "beam        seed " << seed << "  direction ["
              << direction[0] << ", " << direction[1] << ", " << direction[2]
              << "]  distance " << distance[0] << " m\n";

    // The encoder, loaded once and kept. A frame re-encodes; it never rebuilds.
    BeamEncoder encoder(package, kEncoderSpectrumBins, kLatentWidth);
    encoder.set_spectrum(spectrum);
    encoder.set_distance(distance[0]);

    // One latent describes the BEAM, not a point in space, so it is replicated across the queries.
    // A composition would do exactly this with `Invocation::Once` and a repeat, in device memory.
    std::vector<float> latents(std::size_t(queries) * kLatentWidth);
    session->bind_input("latent", memory::host::MemoryRef::of(std::span<float>(latents)));

    std::vector<float> region_state = encode_region(package, options.region_width);
    std::cout << "region      width " << options.region_width << " -> region_state["
              << region_state.size() << "]\n";
    session->bind_input("region_state", memory::host::MemoryRef::of(std::span<float>(region_state)));

    // THE binding that matters: the model's flux output IS the renderer's allocation.
    session->bind_output("flux", flux);
    // `spectrum` is produced too and must go somewhere, but nothing here displays it.
    std::vector<float> spectrum_out(std::size_t(queries) * 32);
    session->bind_output("spectrum", memory::host::MemoryRef::of(std::span<float>(spectrum_out)));

    // Voxel centres of the unit box, in RadFiled3D's own order.
    std::vector<float> positions(std::size_t(queries) * 3);
    for (std::uint32_t z = 0; z < side; ++z)
        for (std::uint32_t y = 0; y < side; ++y)
            for (std::uint32_t x = 0; x < side; ++x) {
                const std::size_t i = (std::size_t(z) * side + y) * side + x;
                positions[i * 3 + 0] = (float(x) + 0.5f) / float(side);
                positions[i * 3 + 1] = (float(y) + 0.5f) / float(side);
                positions[i * 3 + 2] = (float(z) + 0.5f) / float(side);
            }
    session->bind_input("position", memory::host::MemoryRef::of(std::span<float>(positions)));

    // Encode the tube, fan the latent out over the queries, and run the trunk straight into the
    // renderer's allocation. The only thing that changes between frames is the direction.
    const auto run_tube = [&](const std::array<float, 3>& direction_now) {
        const std::vector<float>& latent = encoder.encode(direction_now);
        for (std::int64_t q = 0; q < queries; ++q)
            std::copy(latent.begin(), latent.end(), latents.begin() + q * std::int64_t(latent.size()));
        if (std::getenv("RFNN_REBIND"))
            session->bind_input("latent", memory::host::MemoryRef::of(std::span<float>(latents)));
        session->infer();
        // THE TWO APIS SHARE MEMORY BUT NOT A TIMELINE. `infer()` returns once the work is
        // SUBMITTED; the writes land when the CUDA stream drains. Reading the allocation from
        // Vulkan — or from the host, as the windowing below does — before then shows a volume that
        // is half the previous beam and half the new one, which on screen looks like two beams
        // overlapping.
        //
        // A shipping renderer would share a SEMAPHORE as well as the memory: export a VkSemaphore,
        // import it with `cudaImportExternalSemaphore`, signal it after the inference and have the
        // graphics queue wait on it — no host round trip and no stall. A full device sync is the
        // blunt version, and it is the honest one for a single-threaded example: correct, obvious,
        // and it makes the cost of not having the semaphore visible in the frame time.
        if (cudaDeviceSynchronize() != cudaSuccess)
            die("cudaDeviceSynchronize failed after inference");
    };

    // THE SWEEP HAS ITS OWN ELEVATION, and does not inherit the random beam's.
    //
    // Taking `asin(direction.y)` from a randomly drawn direction looks reasonable and is not: a
    // direction sampled uniformly on the sphere is usually nowhere near the horizon, and this one
    // came out at y = -0.94. The resulting "circle" is a tiny cone around the pole — the beam stays
    // pointed almost straight down for the whole revolution and the field barely moves, which reads
    // as a sweep that does nothing. A C-arm travels in a plane around the patient, so the sweep is
    // given a modest elevation of its own.
    const float tilt = options.tilt;
    float tube_angle = std::atan2(direction[2], direction[0]);
    run_tube(tube_direction(tube_angle, tilt));
    std::cout << "sweep       elevation " << tilt << " rad, one revolution every 30 s\n"
              << "inference   " << queries << " voxels written into the renderer's memory\n";

    // ── 4. what Vulkan sees in its own allocation ───────────────────────────────────────────
    const std::vector<float> seen = renderer.read_volume();
    double sum = 0.0, peak = 0.0;
    std::size_t nonzero = 0;
    for (float v : seen) {
        if (v > 0.f) ++nonzero;
        sum += v;
        peak = std::max(peak, double(v));
    }
    std::size_t peak_at = 0;
    float lowest = seen[0];
    for (std::size_t i = 0; i < seen.size(); ++i) {
        if (seen[i] > seen[peak_at]) peak_at = i;
        lowest = std::min(lowest, seen[i]);
    }
    std::cout << "volume      " << nonzero << "/" << seen.size() << " voxels non-zero, peak "
              << peak << ", mean " << (sum / double(seen.size())) << "\n"
              << "            dynamic range " << (peak / std::max(double(lowest), 1e-30)) << "x, peak at voxel ("
              << (peak_at % side) << ", " << ((peak_at / side) % side) << ", " << (peak_at / (side * side))
              << ") of " << side << "\n";
    if (nonzero == 0) die("the shared allocation is empty — inference did not reach the renderer");

    // ── 5. ray-march the same buffer ────────────────────────────────────────────────────────
    Renderer::View view{0.9f, 0.45f, 2.1f, 0.f, 0.f};
    window_for(seen, view);

    if (options.frames > 0) {
        // The same sweep the window shows, written out frame by frame — so what the animation does
        // can be inspected without watching it.
        for (std::uint32_t f = 0; f < options.frames; ++f) {
            const float angle = tube_angle + 2.f * std::numbers::pi_v<float> * float(f) / float(options.frames);
            const auto dir = tube_direction(angle, tilt);
            if (std::getenv("RFNN_WIPE")) renderer.zero_volume();
            run_tube(dir);
            if (std::getenv("RFNN_ZERO_LATENT") && (f % 2) == 1) {
                std::fill(latents.begin(), latents.end(), 0.f);
                session->infer();
                if (cudaDeviceSynchronize() != cudaSuccess) die("sync");
                std::cout << "        (latent zeroed for this frame)\n";
            }
            const std::vector<float> volume = renderer.read_volume();
            window_for(volume, view);
            // Where the field actually peaks, and how much is near that peak: a second beam would
            // show as a large fraction of bright voxels far from the first.
            std::size_t peak_at = 0;
            for (std::size_t i = 0; i < volume.size(); ++i) if (volume[i] > volume[peak_at]) peak_at = i;
            const float threshold = volume[peak_at] * 0.5f;
            std::size_t hot = 0, hot_far = 0;
            const auto coord = [side](std::size_t i) {
                return std::array<int, 3>{int(i % side), int((i / side) % side), int(i / (side * side))};
            };
            const auto pc = coord(peak_at);
            for (std::size_t i = 0; i < volume.size(); ++i) {
                if (volume[i] < threshold) continue;
                ++hot;
                const auto c = coord(i);
                const int d2 = (c[0]-pc[0])*(c[0]-pc[0]) + (c[1]-pc[1])*(c[1]-pc[1]) + (c[2]-pc[2])*(c[2]-pc[2]);
                if (d2 > int(side * side) / 4) ++hot_far;   // more than half a cube away from the peak
            }
            std::cout << "frame " << f << "  dir [" << dir[0] << ", " << dir[1] << ", " << dir[2]
                      << "]  peak (" << pc[0] << "," << pc[1] << "," << pc[2] << ")  hot " << hot
                      << "  hot-far-from-peak " << hot_far << "\n";
            if (!options.out.empty()) {
                const std::vector<std::uint32_t> pixels = renderer.render(view);
                write_ppm(options.out + "." + std::to_string(f) + ".ppm", pixels, options.width,
                          options.height);
            }
        }
        return 0;
    }

    if (!options.present) {
        const std::vector<std::uint32_t> pixels = renderer.render(view);
        const std::uint32_t background = pixels[0];
        std::size_t lit = 0;
        for (std::uint32_t p : pixels) if (p != background) ++lit;
        std::cout << "raymarch    " << options.width << "x" << options.height << ", " << lit
                  << " pixels above background\n";
        if (lit == 0) die("the render is empty — the shader saw nothing in the shared buffer");
        if (!options.out.empty()) {
            write_ppm(options.out, pixels, options.width, options.height);
            std::cout << "wrote       " << options.out << "\n";
        }
        return 0;
    }

    std::cout << "\ncontrols    drag to orbit the CAMERA · wheel to zoom · O pause/resume the tube sweep · "
                 "N new tube · R reset view · Q quit\n";
    const Renderer::View home = view;
    Renderer::Input input;
    // A slow turn, about half a minute to the revolution, so the beam's shape reads in three
    // dimensions rather than as one flat projection. Advanced by WALL-CLOCK time, not per frame:
    // the ray-march costs whatever the grid costs, and a per-frame step would spin at a speed that
    // depended on the resolution.
    // A slow sweep of the TUBE around the field, the way a C-arm travels: one full circle every 30
    // seconds. Advanced by WALL-CLOCK time rather than per frame, because a frame costs whatever the
    // grid costs and a per-frame step would sweep at a speed that depended on the resolution — the
    // revolution has to take 30 seconds at 48 voxels and at 96. The camera stays where the user put
    // it, so what moves on screen is the field itself.
    constexpr float kSweepSeconds = 30.f;
    constexpr float kSweepRadiansPerSecond = 2.f * std::numbers::pi_v<float> / kSweepSeconds;
    bool sweeping = true;
    auto previous = std::chrono::steady_clock::now();
    while (renderer.poll(view, input)) {
        const auto now = std::chrono::steady_clock::now();
        const float dt = std::chrono::duration<float>(now - previous).count();
        previous = now;
        if (input.toggle_orbit) {
            sweeping = !sweeping;
            std::cout << "sweep       " << (sweeping ? "on" : "paused") << "\n";
        }
        if (input.reset) { view.yaw = home.yaw; view.pitch = home.pitch; view.distance = home.distance; }
        if (input.reroll) {
            // A different tube: new distance and spectrum, same sweep. The allocation is untouched.
            const Beam fresh = roll_beam(rng);
            encoder.set_spectrum(fresh.spectrum);
            encoder.set_distance(fresh.distance[0]);
            std::cout << "tube        distance " << fresh.distance[0] << " m, new spectrum\n";
        }

        if (sweeping || input.reroll) {
            tube_angle += sweeping ? kSweepRadiansPerSecond * dt : 0.f;
            run_tube(tube_direction(tube_angle, tilt));
            // The field's range travels with the beam, so the ramp is re-windowed each frame.
            // Reading the volume back through Vulkan is what proves, every single frame, that the
            // shader and the network are looking at the same bytes.
            window_for(renderer.read_volume(), view);
        }
        renderer.present(view);
    }
    std::cout << "closed\n";
    return 0;
} catch (const std::exception& err) {
    std::cerr << "vk_raymarch: " << err.what() << "\n";
    return 1;
}
