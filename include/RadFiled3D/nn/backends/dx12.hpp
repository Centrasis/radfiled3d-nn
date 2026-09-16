// `rfnn::dx12` — Direct3D 12 interop, compiled in with RFNN_WITH_DX12.
//
// Its own option, not a shared "DirectX" one: D3D11 and D3D12 are separate backends with separate
// import types, handle kinds and dedicated rules (see memory/dx12.hpp), so a build that wants one
// should not carry the other.
//
// Like Vulkan, this needs no SDK and no Windows: the import is a CUDA/HIP call on a handle the
// renderer exported. The probe is declared unconditionally so a build without the option answers
// `false` instead of failing to link.
#pragma once

namespace RadFiled3D::nn::dx12 {

/// Non-zero when this build has the D3D12 interop compiled in.
bool available() noexcept;

}  // namespace RadFiled3D::nn::dx12
