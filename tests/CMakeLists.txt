# GoogleTest, fetched — the same framework RadFiled3D uses for its own suite.
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0
  GIT_SHALLOW    TRUE
  EXCLUDE_FROM_ALL
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# The format's guard rail. Links rfnn::deploy ONLY — a test that needed RadFiled3D here would be a
# sign that the container had grown a native dependency (R-B4).
add_executable(rf3m_roundtrip rf3m_roundtrip.cpp)
set_target_properties(rf3m_roundtrip PROPERTIES FOLDER "tests")
target_link_libraries(rf3m_roundtrip PRIVATE rfnn::deploy GTest::gtest_main)
gtest_discover_tests(rf3m_roundtrip)

add_executable(radfiled3d_boundary radfiled3d_boundary.cpp)
set_target_properties(radfiled3d_boundary PROPERTIES FOLDER "tests")
target_link_libraries(radfiled3d_boundary PRIVATE rfnn::rfnn GTest::gtest_main)
# The test asserts that each probe answers what the build was configured with. ONNX is absent from
# this list deliberately: it has no option any more, so there is nothing for the test to mirror.
foreach(backend CUDA VULKAN DX11 DX12)
  if(RFNN_WITH_${backend})
    target_compile_definitions(radfiled3d_boundary PRIVATE RFNN_TEST_WITH_${backend})
  endif()
endforeach()
gtest_discover_tests(radfiled3d_boundary)

# Drives the C ABI through the shared library, the way a plugin does.
# Filling a field from a session. Links rfnn::rfnn for the field types and the ONNX backend; the
# cases skip without ONNX Runtime, so the target builds in every configuration.
add_executable(field_inference field_inference.cpp)
set_target_properties(field_inference PROPERTIES FOLDER "tests")
target_link_libraries(field_inference PRIVATE rfnn::rfnn GTest::gtest_main)
gtest_discover_tests(field_inference)

add_executable(c_api c_api.cpp)
set_target_properties(c_api PROPERTIES FOLDER "tests")
target_link_libraries(c_api PRIVATE rfnn::c rfnn::deploy GTest::gtest_main)
gtest_discover_tests(c_api)

# Vulkan → CUDA interop. THE TEST links the Vulkan loader; the library deliberately does not (see
# the note on RFNN_WITH_VULKAN in the top-level CMakeLists) — this binary plays the renderer, which
# is the one role that really needs Vulkan. Built only when both halves of the pairing are compiled
# in and a loader is actually installed; the cases skip at run time without a GPU.
if(RFNN_WITH_VULKAN AND RFNN_WITH_CUDA)
  find_package(Vulkan QUIET)
  if(Vulkan_FOUND)
    add_executable(vulkan_interop vulkan_interop.cpp)
    set_target_properties(vulkan_interop PROPERTIES FOLDER "tests")
    target_link_libraries(vulkan_interop PRIVATE rfnn::rfnn Vulkan::Vulkan CUDA::cudart GTest::gtest_main)
    gtest_discover_tests(vulkan_interop)
  else()
    message(STATUS "Vulkan loader not found; the Vulkan interop test will not be built")
  endif()
endif()

# The D3D12 path. Builds and runs anywhere — a shared NT handle only exists on Windows, so the cases
# pin down the surrounding behaviour (pairings, refusals, the resource/heap distinction) and skip the
# import itself. Needs no Direct3D SDK, for the same reason the library does not.
if(RFNN_WITH_DX12)
  add_executable(d3d12_interop d3d12_interop.cpp)
  set_target_properties(d3d12_interop PROPERTIES FOLDER "tests")
  target_link_libraries(d3d12_interop PRIVATE rfnn::rfnn GTest::gtest_main)
  gtest_discover_tests(d3d12_interop)
endif()

if(RFNN_WITH_DX11)
  add_executable(d3d11_interop d3d11_interop.cpp)
  set_target_properties(d3d11_interop PROPERTIES FOLDER "tests")
  target_link_libraries(d3d11_interop PRIVATE rfnn::rfnn GTest::gtest_main)
  gtest_discover_tests(d3d11_interop)
endif()
