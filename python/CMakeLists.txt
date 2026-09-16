# The pybind11 extension module `RadFiled3D.nn._rfnn`.
#
# It installs INTO RadFiled3D's own package directory, which is what makes
# `import RadFiled3D.nn` work: RadFiled3D ships a regular package (it has an
# `__init__.py`), so its `__path__` is one directory and a subpackage has to live in it.
# Two distributions writing into one directory is deliberate here and is why
# `RadFiled3D` is a hard runtime dependency rather than an optional one.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# ── pybind11: THE ONE RadFiled3D USES ────────────────────────────────────────────────────────────
#
# Not a detail. pybind11 keeps its type registry in per-process "internals" keyed by its version and
# ABI, and two extension modules only share registered types when those keys agree. That sharing is
# what lets `GPUCartesianRadiationField.to_host_field()` hand back RadFiled3D's OWN
# `CartesianRadiationField` instead of a wrapper this module would otherwise have to invent. A
# mismatch is not a build error: it surfaces as a cast failure the first time a field crosses.
#
# So the version is never chosen here. In order:
#   1. a pybind11 target already resolved in this build tree — reuse it outright;
#   2. one the environment provides (scikit-build-core puts the build requirement's CMake config on
#      CMAKE_PREFIX_PATH, which is how the two wheels end up on the same one);
#   3. otherwise fetch the tag read out of RadFiled3D's OWN CMakeLists, so the pin lives in one
#      place — its place — and bumping it there carries here without an edit.
#
# RadFiled3D's pin, read out of its own CMakeLists so it exists in one place — its place.
set(_pybind11_tag "")
if(RFNN_RADFILED3D_SOURCE_DIR AND EXISTS "${RFNN_RADFILED3D_SOURCE_DIR}/CMakeLists.txt")
  file(READ "${RFNN_RADFILED3D_SOURCE_DIR}/CMakeLists.txt" _rf3_cmake)
  # Its Declare block: GIT_REPOSITORY .../pybind11.git followed by GIT_TAG <tag>.
  string(REGEX MATCH "pybind11\\.git[ \t\r\n]+GIT_TAG[ \t]+([^ \t\r\n#]+)" _m "${_rf3_cmake}")
  set(_pybind11_tag "${CMAKE_MATCH_1}")
endif()

if(TARGET pybind11::module)
  message(STATUS "pybind11: reusing the target already present in this build tree")
else()
  find_package(pybind11 CONFIG QUIET)
  if(pybind11_FOUND)
    message(STATUS "pybind11: using ${pybind11_VERSION} from the environment")
    # A found pybind11 is not necessarily RadFiled3D's. Say so rather than let it surface later as
    # `to_host_field()` refusing to convert, which reads like a bug in the binding and is not one.
    if(_pybind11_tag AND NOT "v${pybind11_VERSION}" STREQUAL "${_pybind11_tag}")
      message(WARNING
        "pybind11 ${pybind11_VERSION} does not match RadFiled3D's pin ${_pybind11_tag}. The two "
        "modules share pybind11's type registry only when they agree, so GPUCartesianRadiationField."
        "to_host_field() may not convert to RadFiled3D's CartesianRadiationField. store_rf3() is "
        "unaffected.")
    endif()
  else()
    if(NOT _pybind11_tag)
      message(FATAL_ERROR
        "Could not read RadFiled3D's pybind11 pin from '${RFNN_RADFILED3D_SOURCE_DIR}'. This module "
        "must use the same pybind11 as RadFiled3D or the two stop sharing types. Install a matching "
        "pybind11 so find_package(pybind11) succeeds, or build with -DRFNN_BUILD_PYTHON=OFF.")
    endif()
    message(STATUS "pybind11: fetching ${_pybind11_tag} (RadFiled3D's pin)")
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        ${_pybind11_tag}
      GIT_SHALLOW    TRUE
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(pybind11)
  endif()
endif()

pybind11_add_module(_rfnn bindings/bindings.cpp)
target_link_libraries(_rfnn PRIVATE rfnn::rfnn)
# The vendored ONNX Runtime sits beside the module in the installed package.
set_target_properties(_rfnn PROPERTIES INSTALL_RPATH "$ORIGIN")

# ── stubs: generated from the built module, never hand-written ───────────────────────────────────
# pybind11-stubgen imports the module and writes the stubs from the signatures and docstrings
# pybind11 recorded, so they cannot drift. It runs after the link, against the build tree.
#
# A module with SUBMODULES stubs as a PACKAGE — `_rfnn/__init__.pyi` plus one `.pyi` per submodule —
# not as a single `_rfnn.pyi`. The whole tree is installed for that reason; naming one file was what
# the single-module layout allowed and it silently stopped matching when `deploy`, `inference` and
# `memory` appeared.
option(RFNN_PYTHON_STUBS "Generate the .pyi stubs with pybind11-stubgen after the build" ON)
if(RFNN_PYTHON_STUBS)
  execute_process(COMMAND "${Python_EXECUTABLE}" -c "import pybind11_stubgen"
                  RESULT_VARIABLE _stubgen_missing OUTPUT_QUIET ERROR_QUIET)
  if(_stubgen_missing)
    message(WARNING "pybind11-stubgen is not importable by ${Python_EXECUTABLE}; no .pyi will be generated "
                    "(pip install pybind11-stubgen, or -DRFNN_PYTHON_STUBS=OFF)")
  else()
    set(_stub_dir "${CMAKE_CURRENT_BINARY_DIR}/stubs")
    add_custom_command(TARGET _rfnn POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=$<TARGET_FILE_DIR:_rfnn>"
              "${Python_EXECUTABLE}" -m pybind11_stubgen --output-dir "${_stub_dir}" --exit-code _rfnn
      COMMENT "Generating the _rfnn stubs from the built module"
      VERBATIM)
    install(DIRECTORY "${_stub_dir}/" DESTINATION RadFiled3D/nn)
  endif()
endif()

# ── install into the package ─────────────────────────────────────────────────────────────────────
install(TARGETS _rfnn LIBRARY DESTINATION RadFiled3D/nn RUNTIME DESTINATION RadFiled3D/nn)
if(RFNN_WITH_ONNX)
  # What the loader opens, and nothing more (see cmake/OnnxRuntime.cmake).
  install(FILES ${RFNN_ORT_RUNTIME_FILES} DESTINATION RadFiled3D/nn)
  if(RFNN_ORT_SONAME_FILE)
    install(FILES "${RFNN_ORT_SONAME_FILE}" DESTINATION RadFiled3D/nn RENAME "${RFNN_ORT_SONAME_NAME}")
  endif()
endif()
