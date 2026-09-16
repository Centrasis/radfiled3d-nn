# ONNX Runtime — fetched on demand (requirements.md R-B5), so nobody installs an ORT SDK by hand and
# the wheel can ship a matching runtime. -DORT_ROOT=<path> uses an existing installation instead.
#
# TWO sources, because Microsoft ships the providers in two places:
#   * GitHub releases — the CPU and CUDA/TensorRT archives. Checked: v1.29.0 publishes
#     linux-x64, linux-x64-gpu_cuda12/13, win-x64, win-x64-gpu_cuda12/13 and the arm variants, and
#     NO DirectML asset at all.
#   * NuGet — `Microsoft.ML.OnnxRuntime.DirectML`, which is where the DirectML provider lives. A
#     .nupkg is a zip, so FetchContent takes it directly. It is the only way to get
#     `dml_provider_factory.h` (and therefore `OrtDmlApi`), which the release archives omit.
#
# ROCm remains ORT_ROOT-only: it is published as source or an AMD-hosted wheel, neither fetchable.
#
# Defines the imported target `onnxruntime` and the list RFNN_ORT_RUNTIME_FILES: the files a
# consumer must ship beside its binary (the SONAME and the provider libraries — never the whole
# archive, whose symlinks install(FILES) would dereference into three copies).
set(ORT_VERSION "1.29.0" CACHE STRING "ONNX Runtime release to fetch")
set(ORT_ROOT "" CACHE PATH "Existing ONNX Runtime root (include/ + lib/); overrides the download")
set(ORT_CUDA "13" CACHE STRING "CUDA major of the ONNX Runtime GPU build: 13 or 12")
# DirectML tracks its own ORT version: the NuGet package lags the GitHub releases (1.24.4 against
# 1.29.0 at time of writing), so it cannot share ORT_VERSION.
set(ORT_DML_VERSION "1.24.4" CACHE STRING "Microsoft.ML.OnnxRuntime.DirectML NuGet version")
# The GPU package is fetched exactly when a CUDA execution provider was asked for. It is ~180 MB
# against ~10 MB for the CPU build, so a CPU-only configuration must not pay for it.
option(ORT_GPU "Fetch the CUDA/TensorRT ONNX Runtime build" ${RFNN_WITH_CUDA})

# One build links ONE ONNX Runtime, and no published package carries both the CUDA/TensorRT and the
# DirectML providers — they come from different sources at different versions. Asking for both would
# silently get whichever branch ran first.
if(RFNN_WITH_DIRECTML AND RFNN_WITH_CUDA AND NOT ORT_ROOT)
  message(FATAL_ERROR
    "RFNN_WITH_DIRECTML and RFNN_WITH_CUDA need different ONNX Runtime packages (NuGet DirectML vs "
    "the GitHub CUDA archive) and a build links only one. Enable one, or point -DORT_ROOT=<path> at "
    "a runtime you built with both providers.")
endif()

if(ORT_ROOT)
  message(STATUS "ONNX Runtime: using ${ORT_ROOT}")
  set(_ORT_INCLUDE_DIR "${ORT_ROOT}/include")
  set(_ORT_LIB_DIR "${ORT_ROOT}/lib")
elseif(RFNN_WITH_ROCM)
  # Published as source or an AMD-hosted wheel; there is no archive to fetch.
  message(FATAL_ERROR
    "The ROCm execution provider has no prebuilt ONNX Runtime package. Build or install one and "
    "point -DORT_ROOT=<path> at it (a directory with include/ and lib/).")
elseif(RFNN_WITH_DIRECTML)
  # Windows-only by construction: RFNN_WITH_DIRECTML is gated on WIN32 in the top-level CMakeLists,
  # so reaching here means Windows and the option on — which is exactly when this fetch should run
  # and never otherwise.
  if(NOT WIN32)
    message(FATAL_ERROR "The DirectML ONNX Runtime package is Windows-only.")
  endif()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|arm64|aarch64")
    set(_ORT_DML_ARCH "win-arm64")
  else()
    set(_ORT_DML_ARCH "win-x64")
  endif()
  message(STATUS "ONNX Runtime: fetching Microsoft.ML.OnnxRuntime.DirectML ${ORT_DML_VERSION} (NuGet)")
  include(FetchContent)
  FetchContent_Declare(
    fetch_onnxruntime_dml
    URL https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/${ORT_DML_VERSION}/microsoft.ml.onnxruntime.directml.${ORT_DML_VERSION}.nupkg
  )
  FetchContent_MakeAvailable(fetch_onnxruntime_dml)
  # A .nupkg lays its files out differently from a release archive: headers under build/native,
  # binaries under runtimes/<arch>/native. Hence the separate include/lib variables.
  set(_ORT_INCLUDE_DIR "${fetch_onnxruntime_dml_SOURCE_DIR}/build/native/include")
  set(_ORT_LIB_DIR "${fetch_onnxruntime_dml_SOURCE_DIR}/runtimes/${_ORT_DML_ARCH}/native")
else()
  if(WIN32)
    set(_ORT_OS "win-x64")
    set(_ORT_EXT "zip")
  else()
    set(_ORT_OS "linux-x64")
    set(_ORT_EXT "tgz")
  endif()
  if(ORT_GPU)
    if(ORT_CUDA STREQUAL "13")
      set(_ORT_VARIANT "-gpu_cuda13")
    elseif(ORT_VERSION VERSION_LESS "1.27.0")
      # Up to 1.26 the CUDA-12 build was the unsuffixed `-gpu` asset; from 1.27 it is explicit.
      set(_ORT_VARIANT "-gpu")
    else()
      set(_ORT_VARIANT "-gpu_cuda12")
    endif()
  else()
    set(_ORT_VARIANT "")
  endif()
  set(_ORT_PKG "onnxruntime-${_ORT_OS}${_ORT_VARIANT}-${ORT_VERSION}")
  message(STATUS "ONNX Runtime: fetching ${_ORT_PKG}")
  include(FetchContent)
  FetchContent_Declare(
    fetch_onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${_ORT_PKG}.${_ORT_EXT}
  )
  FetchContent_MakeAvailable(fetch_onnxruntime)
  set(_ORT_INCLUDE_DIR "${fetch_onnxruntime_SOURCE_DIR}/include")
  set(_ORT_LIB_DIR "${fetch_onnxruntime_SOURCE_DIR}/lib")
endif()

# A prebuilt ORT is a shared library: an IMPORTED target, never compiled here.
add_library(onnxruntime SHARED IMPORTED GLOBAL)
if(WIN32)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION             "${_ORT_LIB_DIR}/onnxruntime.dll"
    IMPORTED_IMPLIB               "${_ORT_LIB_DIR}/onnxruntime.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${_ORT_INCLUDE_DIR}")
  # Windows ships plain DLLs — no symlinks, nothing to resolve.
  file(GLOB RFNN_ORT_RUNTIME_FILES "${_ORT_LIB_DIR}/*.dll")
  set(RFNN_ORT_SONAME_FILE "")
  set(RFNN_ORT_SONAME_NAME "")
else()
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION             "${_ORT_LIB_DIR}/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${_ORT_INCLUDE_DIR}")

  # The archive ships libonnxruntime.so -> .so.1 -> .so.1.29.0, and only the SONAME (.so.1) is
  # opened at load time. `install(FILES)` copies a symlink AS A SYMLINK, so installing `.so.1`
  # directly drops a DANGLING link beside the module — which a wheel then silently omits. Resolve it
  # here and install the real file under the SONAME's name instead (see the RENAME at each install
  # site). Shipping the whole lib/ directory is not the answer either: that is the same 28 MB three
  # times over.
  file(GLOB _ort_soname_link "${_ORT_LIB_DIR}/libonnxruntime.so.[0-9]")
  if(NOT _ort_soname_link)
    message(FATAL_ERROR "No libonnxruntime.so.<N> in ${_ORT_LIB_DIR}; the archive layout changed.")
  endif()
  file(REAL_PATH "${_ort_soname_link}" RFNN_ORT_SONAME_FILE)
  get_filename_component(RFNN_ORT_SONAME_NAME "${_ort_soname_link}" NAME)
  # The execution-provider libraries are plain files and need no resolving.
  file(GLOB RFNN_ORT_RUNTIME_FILES "${_ORT_LIB_DIR}/libonnxruntime_providers_*.so")
  message(STATUS "ONNX Runtime: will ship ${RFNN_ORT_SONAME_NAME} (from ${RFNN_ORT_SONAME_FILE})")
endif()
set(RFNN_ORT_LIB_DIR "${_ORT_LIB_DIR}")
