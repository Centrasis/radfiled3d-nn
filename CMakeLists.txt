# radfiled3d-nn — RF3M: the deployment container and GPU inference runtime for neural radiation
# fields. C++20, CMake ≥ 3.24. The whole native dependency chain lives here so that a fresh clone
# builds with nothing but CMake and a C++20 compiler (requirements.md R-B2).
#
# Targets
#   rfnn::deploy   the .rf3m container. Links BLAKE3 and nothing else — that is R-B4, enforced by
#                  the build rather than by a comment.
#   rfnn::rfnn     the library: fields, sessions, memory interop, backends. Links RadFiled3D.
#   rfnn::c        the C ABI as a shared library, the surface a prebuilt-binary consumer (an
#                  Unreal Engine plugin) links against. The only installed/exported target.
#   rf3m           the inspection tool (links rfnn::deploy only)
#   _rfnn          the pybind11 extension module, with RFNN_BUILD_PYTHON
#
# Options
#   RadFiled3D is mandatory and has no option (R-B1). The options are exactly the accelerator
#   backends (R-B3), each mapping 1:1 to a compile define of the same name; a disabled backend is
#   compiled out and the API answers with FeatureDisabled.
cmake_minimum_required(VERSION 3.24)
# ── the version ──────────────────────────────────────────────────────────────────────────────────
#
# ONE number for the wheel and for the C++ library inside it, and the RELEASE TAG is where it comes
# from. It flows into `include/RadFiled3D/nn/version.hpp.in`, and so into `kVersion`, the C ABI's
# `rfnn_version()` and Python's `RadFiled3D.nn.version()`.
#
# Resolved in this order, most explicit first:
#
#   -DRFNN_VERSION=X.Y.Z        what the CI passes to a plain cmake build
#   SKBUILD_PROJECT_VERSION     what scikit-build-core passes during a wheel build, taken from
#                               build_tools/radfiled3d_nn_version.py — which is what makes the wheel
#                               and this library agree BY CONSTRUCTION rather than by a check
#   CI_COMMIT_TAG / GITHUB_REF  the tag, for a CI build that went through neither of the above
#   0.0.0                       a working tree that is not a release, which is the honest answer
set(RFNN_VERSION "" CACHE STRING "Version to build as; empty resolves it from the CI tag")
set(_rfnn_version "${RFNN_VERSION}")
if(NOT _rfnn_version AND DEFINED SKBUILD_PROJECT_VERSION)
  set(_rfnn_version "${SKBUILD_PROJECT_VERSION}")
endif()
if(NOT _rfnn_version AND DEFINED ENV{CI_COMMIT_TAG})
  set(_rfnn_version "$ENV{CI_COMMIT_TAG}")
endif()
if(NOT _rfnn_version AND DEFINED ENV{GITHUB_REF})
  string(REGEX REPLACE "^.*/" "" _rfnn_version "$ENV{GITHUB_REF}")
endif()
# A branch name is not a version, and neither is a tag that is not X.Y.Z.
if(NOT _rfnn_version MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+")
  set(_rfnn_version "0.0.0")
endif()
# `project()` wants bare X.Y.Z, so a suffix like `1.2.3rc1` is trimmed for CMake and kept nowhere:
# the wheel carries the full string, the library carries the release it was cut from.
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" _rfnn_version "${_rfnn_version}")

project(radfiled3d_nn VERSION ${_rfnn_version} LANGUAGES CXX C)
message(STATUS "radfiled3d-nn version ${PROJECT_VERSION}")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Everything static is linked into a shared library (rfnn::c, the Python module), so all of it is
# position-independent, RadFiled3D and BLAKE3 included.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(RFNN_WITH_CUDA     "Enable the CUDA execution provider"                   OFF)
option(RFNN_WITH_TENSORRT "Enable the TensorRT execution provider"               OFF)
option(RFNN_WITH_ROCM     "Enable the ROCm/MIGraphX execution provider (AMD)"    OFF)
option(RFNN_WITH_DIRECTML "Enable the DirectML execution provider (Windows)"     OFF)
option(RFNN_WITH_VULKAN   "Enable Vulkan external-memory interop"                OFF)
option(RFNN_WITH_DX11     "Enable Direct3D 11 external-memory interop"           OFF)
option(RFNN_WITH_DX12     "Enable Direct3D 12 external-memory interop"           OFF)
# The CPU execution provider. Its own option like every other backend, though it can only be
# offered when ONNX Runtime is present, since it IS ORT's provider.
option(RFNN_WITH_CPU      "Enable the portable CPU execution provider"           ON)

# The pybind11 bindings are a real consumer of the C++ API: a rename that misses bindings.cpp has
# to fail in the ordinary build, not later in `pip install`. So they are ON by default for a
# top-level build whenever a Python development environment is present — and OFF, with a status
# line, when it is not, which keeps R-B2 true: CMake and a C++20 compiler are still the only
# prerequisites for the library, the tool and the tests.
set(RFNN_BUILD_PYTHON_DEFAULT OFF)

# Set CMAKE_PREFIX_PATH for Conda environment
# This allows CMake to find Python development components in conda environments
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
  list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/bin")
  list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/lib")
  message(STATUS "CMAKE_PREFIX_PATH: $ENV{CONDA_PREFIX}")
endif()
if(DEFINED ENV{CONDA_DEFAULT_ENV})
  message(STATUS "Conda environment: $ENV{CONDA_DEFAULT_ENV}")
  if(DEFINED CONDA_PREFIX)
    message(STATUS "Conda prefix: $CONDA_PREFIX")
  endif()
endif()

if(PROJECT_IS_TOP_LEVEL)
  find_package(Python QUIET COMPONENTS Interpreter Development.Module)
  if(Python_FOUND)
    set(RFNN_BUILD_PYTHON_DEFAULT ON)
  else()
    message(STATUS "Python Development.Module not found; the pybind11 bindings will not be built. "
                   "Point -DPython_EXECUTABLE=<interpreter> at an environment with headers to enable them.")
  endif()
endif()
option(RFNN_BUILD_PYTHON   "Build the pybind11 extension module"     ${RFNN_BUILD_PYTHON_DEFAULT})
option(RFNN_BUILD_TESTS    "Build the GoogleTest suite"               ${PROJECT_IS_TOP_LEVEL})
option(RFNN_BUILD_TOOLS    "Build the rf3m inspection tool"           ${PROJECT_IS_TOP_LEVEL})
option(RFNN_BUILD_EXAMPLES "Build the examples"                       ${PROJECT_IS_TOP_LEVEL})
option(RFNN_INSTALL        "Generate install and CMake package rules" ${PROJECT_IS_TOP_LEVEL})

# The implications the SDKs genuinely impose: TensorRT is a CUDA library, and every execution
# provider needs the ONNX Runtime core. vulkan and directx imply NOTHING (R-G2).
if(RFNN_WITH_TENSORRT)
  # TensorRT is reached through ONNX Runtime's execution provider, never through NvInfer.h: the EP
  # dlopen's libnvinfer at run time, so a TRT build needs no TensorRT SDK — only the GPU ORT archive,
  # which is the one carrying libonnxruntime_providers_{cuda,tensorrt}.so. The CPU archive has
  # neither, silently, so the fetch has to be forced here.
  set(RFNN_WITH_CUDA ON CACHE BOOL "" FORCE)
endif()

if(RFNN_WITH_TENSORRT OR RFNN_WITH_CUDA OR RFNN_WITH_DIRECTML OR RFNN_WITH_DX11 OR RFNN_WITH_DX12 OR RFNN_WITH_ROCM OR RFNN_WITH_VULKAN)
  # if graphics backends are used, we obviously opt for using the GPU ORT variants
  set(ORT_GPU ON CACHE BOOL "" FORCE)
endif()

# DirectML executes on D3D12: its whole advantage is that a D3D12 resource is already its memory, and
# that path is in the D3D12 backend.
if(RFNN_WITH_DIRECTML)
  set(RFNN_WITH_DX12 ON CACHE BOOL "" FORCE)
endif()

configure_file(include/RadFiled3D/nn/version.hpp.in
               "${CMAKE_CURRENT_BINARY_DIR}/generated/RadFiled3D/nn/version.hpp" @ONLY)

# ── the libraries ────────────────────────────────────────────────────────────────────────────────
#
# Three static libraries and one joined target, in a DAG:
#
#     rfnn_deploy      the .rf3m container: loading and storing a model, plus the vocabulary that
#                      description needs (Exception, the geometry types). BLAKE3 + RadFiled3D (glm).
#     rfnn_core        what a model IS and where its memory lives: the `Model` base holding
#                      everything a package describes, the Stage / InferenceSession / ComputeBackend
#                      interfaces, and all of `memory/` — host and device references, and the
#                      graphics interop that imports an engine's allocation.
#     rfnn_inference   how a model RUNS: the concrete backends (ONNX Runtime, CUDA, TensorRT, ROCm,
#                      DirectML) and the field-filling driver.
#     rfnn             all three, joined. What an external consumer and the Python bindings link.
#
# The direction is deploy <- core <- inference and never reverses. `core` holds a `deploy::Package`
# inside `Model`, which is why the container sits BELOW it rather than beside it; the vocabulary
# lives with the container for the same reason, as the lowest thing both sides need.
include(RadFiled3D)
include(Blake3)
include(OnnxRuntime)
set_target_properties(libRadFiled3D PROPERTIES FOLDER "thirdparty")

# ── rfnn::deploy — the container, and the vocabulary under it ────────────────────────────────────
# Never gains ONNX Runtime or a backend: `rf3m` and `rf3m_roundtrip` link this alone, and their link
# line carrying no `libonnxruntime` is what keeps defect D2 fixed.
add_library(rfnn_deploy STATIC
  src/exception.cpp
  src/types.cpp
  src/deploy/codec.cpp
  src/deploy/descriptor.cpp
  src/deploy/composition.cpp
  src/deploy/package.cpp
  src/deploy/version.cpp
  src/deploy/v1.cpp
  src/deploy/builder.cpp)
add_library(rfnn::deploy ALIAS rfnn_deploy)
set_target_properties(rfnn_deploy PROPERTIES FOLDER "Library")
# Declared here and inherited by everything downstream, so the include path is stated once.
target_include_directories(rfnn_deploy PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
  $<INSTALL_INTERFACE:include>)
# RadFiled3D is PUBLIC so that anyone linking any of these targets can use RadFiled3D itself, and so
# that glm — which is RadFiled3D's to fetch — reaches `CartesianFieldGeometry`'s vec3 accessors
# through one edge instead of each target naming it.
target_link_libraries(rfnn_deploy PUBLIC libRadFiled3D PRIVATE BLAKE3::blake3)
target_compile_features(rfnn_deploy PUBLIC cxx_std_20)

# ── rfnn::core — the model and its memory ───────────────────────────────────────────────────────
# No ONNX Runtime and no execution provider here either: this is the interface a backend implements
# and the memory it will be handed, not any way of running anything.
add_library(rfnn_core STATIC
  src/core/model.cpp
  src/core/field.cpp
  src/core/stage_weights.cpp
  src/core/rf3_metadata.cpp
  src/memory/memory_ref.cpp
  src/memory/adopt.cpp
  # CUDA device memory: allocate, copy, and import a handle the renderer exported. Device-and-host
  # memory handling, and it asks nothing of any execution provider, so it belongs here.
  src/memory/cuda.cpp)
add_library(rfnn::core ALIAS rfnn_core)
set_target_properties(rfnn_core PROPERTIES FOLDER "Library")
target_link_libraries(rfnn_core PUBLIC rfnn_deploy)

# ── rfnn::inference — running it ────────────────────────────────────────────────────────────────
add_library(rfnn_inference STATIC
  src/core/session.cpp
  src/core/field_inference.cpp
  src/backends/onnx.cpp
  src/backends/cpu.cpp
  src/backends/cuda.cpp
  src/backends/tensorrt.cpp
  src/backends/rocm.cpp
  src/backends/directml.cpp
  src/backends/vulkan.cpp
  src/backends/dx11.cpp
  src/backends/dx12.cpp)
add_library(rfnn::inference ALIAS rfnn_inference)
set_target_properties(rfnn_inference PROPERTIES FOLDER "Library")
target_link_libraries(rfnn_inference PUBLIC rfnn_core onnxruntime)

# Each enabled backend contributes its define and its libraries. Memory interop is compiled into
# `rfnn_core`, so a define that reaches `src/memory/` has to be set on both targets — noted at each.
# The find_package calls are deliberately REQUIRED: asking for a backend whose SDK is absent must
# fail at configure time with a clear message, not produce a build that silently lacks it.
# A backend's define has to reach BOTH libraries: its memory half is compiled into `rfnn_core`
# (src/memory/) and its compute half into `rfnn_inference` (src/backends/). Setting it on one only
# would compile the import path out while leaving the provider in, or the reverse — a build that
# links but refuses interop at run time for no stated reason.
function(rfnn_backend_define _define)
  target_compile_definitions(rfnn_core PRIVATE ${_define})
  target_compile_definitions(rfnn_inference PRIVATE ${_define})
endfunction()

if(RFNN_WITH_TENSORRT)
  rfnn_backend_define(RFNN_WITH_TENSORRT)
  set(RFNN_WITH_CUDA ON CACHE BOOL "" FORCE)
endif()

if(RFNN_WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(rfnn_core PUBLIC CUDA::cudart CUDA::cuda_driver)
  rfnn_backend_define(RFNN_WITH_CUDA)
endif()

if(RFNN_WITH_ROCM)
  find_package(hip REQUIRED)
  target_link_libraries(rfnn_core PUBLIC hip::host)
  rfnn_backend_define(RFNN_WITH_ROCM)
endif()
if(RFNN_WITH_DIRECTML)
  # DirectML is a Windows API and the option stays Windows-only — unlike RFNN_WITH_DIRECTX, whose
  # import path is pure CUDA and so builds anywhere. Here the provider itself only exists on Windows.
  if(NOT WIN32)
    message(FATAL_ERROR "RFNN_WITH_DIRECTML is Windows-only")
  endif()
  # No d3d12/dxgi link: the only Direct3D thing used is the ID3D12Resource TYPE, to hand a pointer
  # to ORT's CreateGPUAllocationFromD3DResource. The declaration comes from the Windows SDK; no
  # Direct3D function is called, so there is nothing to link.
  #
  # DirectML.dll itself is NOT bundled with the ONNX Runtime NuGet package (checked: it ships only
  # onnxruntime.dll and onnxruntime_providers_shared.dll). Windows 10 1903+ carries a system
  # DirectML.dll that ORT will load; a newer one comes from the Microsoft.AI.DirectML NuGet package
  # under bin/x64-win/. That is a deployment choice, so it is not fetched here — the package is
  # ~200 MB because it carries every architecture including Xbox.
  rfnn_backend_define(RFNN_WITH_DIRECTML)
endif()
if(RFNN_WITH_VULKAN)
  # Deliberately NO find_package(Vulkan): importing a Vulkan allocation is `cudaImportExternalMemory`
  # on a file descriptor or NT handle the renderer already exported, so this library never calls a
  # Vulkan entry point and needs neither the loader nor the SDK. An engine plugin therefore does not
  # have to ship a Vulkan loader on our behalf. (The interop test links the loader — it plays the
  # role of the engine, which is the one role that really does need Vulkan.)
  rfnn_backend_define(RFNN_WITH_VULKAN)
endif()
foreach(_d3d 11 12)
  if(RFNN_WITH_DX${_d3d})
    # Like Vulkan, and for the same reason: importing a Direct3D resource is `cudaImportExternalMemory`
    # on the shared handle the renderer got from D3D. This library never calls a Direct3D entry point,
    # so it links neither d3d11/d3d12 nor dxgi and needs no SDK.
    #
    # It also builds off Windows, deliberately: the code is then compiled and type-checked everywhere
    # instead of rotting behind a platform guard. A shared handle only exists on Windows, so the path
    # is inert elsewhere and says so at run time rather than at configure time.
    if(NOT WIN32)
      message(STATUS "RFNN_WITH_DX${_d3d} on a non-Windows host: the D3D${_d3d} import path is compiled "
                     "but cannot run, since a shared Direct3D handle only exists on Windows.")
    endif()
    rfnn_backend_define(RFNN_WITH_DX${_d3d})
  endif()
endforeach()

if(RFNN_WITH_CPU)
  rfnn_backend_define(RFNN_WITH_CPU)
endif()

# ── rfnn::rfnn — the three, joined ───────────────────────────────────────────────────────────────
# What an external consumer links, and what the Python bindings link. INTERFACE rather than another
# archive: the three libraries above already carry their own objects and usage requirements, so this
# only has to name them. Linking `rfnn::rfnn` gets the container, the runtime and the vocabulary;
# linking `rfnn::deploy` alone still gets just the container.
add_library(rfnn INTERFACE)
add_library(rfnn::rfnn ALIAS rfnn)
target_link_libraries(rfnn INTERFACE rfnn_core rfnn_deploy rfnn_inference)

# ── rfnn::c — the C ABI, as a shared library ─────────────────────────────────────────────────────
# Only the rfnn_* symbols are visible; everything else — RadFiled3D, BLAKE3, the C++ API — is
# linked in statically and hidden, so the binary is self-contained and its ABI is exactly c_api.h.
add_library(rfnn_c SHARED src/capi/c_api.cpp)
add_library(rfnn::c ALIAS rfnn_c)
set_target_properties(rfnn_c PROPERTIES FOLDER "ffi")
target_link_libraries(rfnn_c PRIVATE rfnn)
target_include_directories(rfnn_c PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_definitions(rfnn_c INTERFACE RFNN_SHARED)
set_target_properties(rfnn_c PROPERTIES
  OUTPUT_NAME rfnn_c
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT APPLE)
  # The static archives linked in (BLAKE3's assembly in particular) export default-visibility
  # symbols of their own; this keeps them out of the shared library's dynamic symbol table.
  target_link_options(rfnn_c PRIVATE -Wl,--exclude-libs,ALL)
endif()

# ── tools, examples ──────────────────────────────────────────────────────────────────────────────
if(RFNN_BUILD_TOOLS)
  add_executable(rf3m tools/rf3m.cpp)
  set_target_properties(rf3m PROPERTIES FOLDER "targets")
  target_link_libraries(rf3m PRIVATE rfnn::deploy)
endif()
if(RFNN_BUILD_EXAMPLES)
  add_executable(write_sample examples/write_sample.cpp)
  set_target_properties(write_sample PROPERTIES FOLDER "targets")
  target_link_libraries(write_sample PRIVATE rfnn::deploy)

  # ── vk_raymarch — the interop demonstration ────────────────────────────────────────────────
  # A renderer allocates the volume, inference writes into it, and a compute shader marches the
  # SAME allocation. It needs what a renderer needs and the library deliberately does not: the
  # Vulkan loader, and a shader compiler. Absent either, it is skipped with a status line rather
  # than failing the build — the library itself still needs neither.
  if(RFNN_WITH_VULKAN AND RFNN_WITH_CUDA)
    find_package(Vulkan QUIET)
    find_program(GLSLANG_VALIDATOR NAMES glslangValidator glslang)
    if(Vulkan_FOUND AND GLSLANG_VALIDATOR)
      set(_spv "${CMAKE_CURRENT_BINARY_DIR}/generated/raymarch_spv.h")
      add_custom_command(
        OUTPUT "${_spv}"
        COMMAND "${GLSLANG_VALIDATOR}" -V --vn kRaymarchSpv -o "${_spv}"
                "${CMAKE_CURRENT_SOURCE_DIR}/examples/shaders/raymarch.comp"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/examples/shaders/raymarch.comp"
        COMMENT "Compiling raymarch.comp to SPIR-V"
        VERBATIM)
      add_executable(vk_raymarch examples/vk_raymarch.cpp "${_spv}")
      set_target_properties(vk_raymarch PROPERTIES FOLDER "targets")
      target_include_directories(vk_raymarch PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
      target_link_libraries(vk_raymarch PRIVATE rfnn::rfnn Vulkan::Vulkan)
      # A window, so the field can be explored rather than only dumped to a file. Xlib because its
      # headers are already present and a window plus mouse events is all this needs — under a
      # Wayland session XWayland serves it like any other X client. Without X11, the example still
      # builds and runs with --offscreen.
      # Looked up by hand rather than through find_package(X11): on this machine the headers live in
      # a conda environment while the library is the system one, and FindX11 insists both come from
      # the same prefix. Hint either with -DRFNN_X11_INCLUDE_DIR / -DRFNN_X11_LIBRARY.
      find_path(RFNN_X11_INCLUDE_DIR X11/Xlib.h HINTS "$ENV{CONDA_PREFIX}/include")
      find_library(RFNN_X11_LIBRARY NAMES X11 HINTS "$ENV{CONDA_PREFIX}/lib")
      if(RFNN_X11_INCLUDE_DIR AND RFNN_X11_LIBRARY)
        target_include_directories(vk_raymarch PRIVATE "${RFNN_X11_INCLUDE_DIR}")
        target_link_libraries(vk_raymarch PRIVATE "${RFNN_X11_LIBRARY}")
        message(STATUS "vk_raymarch: window through X11 (${RFNN_X11_LIBRARY})")
      else()
        target_compile_definitions(vk_raymarch PRIVATE RFNN_NO_PRESENT)
        message(STATUS "vk_raymarch: no X11 headers or library, building offscreen-only")
      endif()
    else()
      message(STATUS "vk_raymarch: skipped (needs the Vulkan loader and glslangValidator)")
    endif()
  endif()
endif()

# ── tests ────────────────────────────────────────────────────────────────────────────────────────
if(RFNN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ── python ───────────────────────────────────────────────────────────────────────────────────────
if(RFNN_BUILD_PYTHON)
  add_subdirectory(python)
  set_target_properties(_rfnn PROPERTIES FOLDER "ffi")
endif()

# ── thirdparty packages ────────────────────────────────────────────────────────────────────────────
set_target_properties(blake3 PROPERTIES FOLDER "thirdparty")
set_target_properties(glm PROPERTIES FOLDER "thirdparty")
if(RFNN_BUILD_TESTS)
  set_target_properties(gtest PROPERTIES FOLDER "thirdparty")
  set_target_properties(gtest_main PROPERTIES FOLDER "thirdparty")
endif()

# ── install / package ────────────────────────────────────────────────────────────────────────────
if(RFNN_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  install(TARGETS rfnn_c EXPORT rfnnTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(FILES include/RadFiled3D/nn/c_api.h include/RadFiled3D/nn/c_api.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/RadFiled3D/nn)

  # The runtime the C ABI library opens at load; ships beside it.
  install(FILES ${RFNN_ORT_RUNTIME_FILES} DESTINATION ${CMAKE_INSTALL_LIBDIR})
  if(RFNN_ORT_SONAME_FILE)
    install(FILES "${RFNN_ORT_SONAME_FILE}" DESTINATION ${CMAKE_INSTALL_LIBDIR} RENAME "${RFNN_ORT_SONAME_NAME}")
  endif()
  set_target_properties(rfnn_c PROPERTIES INSTALL_RPATH "$ORIGIN")

  install(EXPORT rfnnTargets NAMESPACE rfnn:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rfnn)
  configure_package_config_file(cmake/rfnnConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/rfnnConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rfnn)
  write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/rfnnConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rfnnConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/rfnnConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rfnn)
endif()
