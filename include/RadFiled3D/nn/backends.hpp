// Every backend's availability probe. One namespace per backend, one file each; a new backend adds
// a file here and never edits a shared one.
#pragma once

#include <RadFiled3D/nn/backends/cuda.hpp>
#include <RadFiled3D/nn/backends/directml.hpp>
#include <RadFiled3D/nn/backends/cpu.hpp>
#include <RadFiled3D/nn/backends/dx11.hpp>
#include <RadFiled3D/nn/backends/dx12.hpp>
#include <RadFiled3D/nn/backends/onnx.hpp>
#include <RadFiled3D/nn/backends/rocm.hpp>
#include <RadFiled3D/nn/backends/tensorrt.hpp>
#include <RadFiled3D/nn/backends/vulkan.hpp>
