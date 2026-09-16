#include <RadFiled3D/nn/backends/dx12.hpp>

namespace RadFiled3D::nn::dx12 {

bool available() noexcept {
#ifdef RFNN_WITH_DX12
    return true;
#else
    return false;
#endif
}

}  // namespace RadFiled3D::nn::dx12
