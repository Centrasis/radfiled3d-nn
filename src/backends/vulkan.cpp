#include <RadFiled3D/nn/backends/vulkan.hpp>

namespace RadFiled3D::nn::vulkan {

bool available() noexcept {
#ifdef RFNN_WITH_VULKAN
    return true;
#else
    return false;
#endif
}

}  // namespace RadFiled3D::nn::vulkan
