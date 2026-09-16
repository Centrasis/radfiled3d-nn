# BLAKE3 — the package digest (requirements.md R-F5).
#
# The only dependency of rfnn::deploy, and the reason it is BLAKE3 rather than a hand-written
# SHA-256: the digest is verified on every load over a payload that can run to hundreds of
# megabytes, and BLAKE3's SIMD paths make that cost negligible where SHA-256 would not. The
# official C implementation is fetched and built as a static library; nothing else of the
# repository is used.
include(FetchContent)
set(BLAKE3_TAG "1.8.2" CACHE STRING "BLAKE3 git tag")
set(BLAKE3_USE_TBB OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  blake3
  GIT_REPOSITORY https://github.com/BLAKE3-team/BLAKE3.git
  GIT_TAG        ${BLAKE3_TAG}
  GIT_SHALLOW    TRUE
  SOURCE_SUBDIR  c
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(blake3)
