# RadFiled3D is a MANDATORY dependency (requirements.md R-B1): this library exists to produce and
# consume its fields, so it is always fetched and always linked, and it in turn fetches glm.
#
# Pinned release. Bumping this is a deliberate act: src/core is written against this version's
# headers. A local checkout is substituted with -DRADFILED3D_SOURCE_DIR=<path>.
set(RADFILED3D_TAG "1.3.6" CACHE STRING "RadFiled3D git tag to build against")
set(RADFILED3D_SOURCE_DIR "" CACHE PATH "Local RadFiled3D checkout; overrides the fetch")

# RadFiled3D defaults its python bindings ON; only the C++ library is wanted here.
set(RS_Build_PyBindings OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS         OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES      OFF CACHE BOOL "" FORCE)

# Where the sources ended up, whichever branch below ran. `python/CMakeLists.txt` reads RadFiled3D's
# own pybind11 pin out of this tree so that pin exists in exactly one place (see there for why a
# mismatch is not a build error but a run-time cast failure).
set(RFNN_RADFILED3D_SOURCE_DIR "" CACHE INTERNAL "RadFiled3D source tree in use")

if(RADFILED3D_SOURCE_DIR)
  message(STATUS "RadFiled3D: using local checkout ${RADFILED3D_SOURCE_DIR}")
  add_subdirectory("${RADFILED3D_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/radfiled3d-build" EXCLUDE_FROM_ALL)
  set(RFNN_RADFILED3D_SOURCE_DIR "${RADFILED3D_SOURCE_DIR}" CACHE INTERNAL "RadFiled3D source tree in use")
else()
  include(FetchContent)
  message(STATUS "RadFiled3D: fetching tag ${RADFILED3D_TAG}")
  FetchContent_Declare(
    radfiled3d
    GIT_REPOSITORY https://github.com/Centrasis/RadFiled3D.git
    GIT_TAG        ${RADFILED3D_TAG}
    GIT_SHALLOW    TRUE
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(radfiled3d)
  set(RFNN_RADFILED3D_SOURCE_DIR "${radfiled3d_SOURCE_DIR}" CACHE INTERNAL "RadFiled3D source tree in use")
endif()
