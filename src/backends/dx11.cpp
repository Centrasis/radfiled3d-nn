#include <RadFiled3D/nn/backends/dx11.hpp>

namespace RadFiled3D::nn::dx11 {

bool available() noexcept {
#ifdef RFNN_WITH_DX11
    return true;
#else
    return false;
#endif
}

}  // namespace RadFiled3D::nn::dx11
