# Download the prebuilt ONNX Runtime arm64 release tarball. These aren't on a
# package registry, so we grab the tgz from the GitHub release.
#
# Updating:
#   - bump ORT_VERSION below
#   - sha256sum -b onnxruntime-osx-arm64-X.Y.Z.tgz
#   - paste into ORT_SHA256
include(FetchContent)

set(ORT_VERSION "1.20.1" CACHE STRING "onnxruntime release tag")
set(ORT_ARCHIVE "onnxruntime-osx-arm64-${ORT_VERSION}")
set(ORT_URL
    "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE}.tgz"
)

# Known-good hash for 1.20.1. Replace on version bump.
set(ORT_SHA256 "" CACHE STRING "sha256 of the ORT tarball; empty = trust the URL")

if(ORT_SHA256)
    FetchContent_Declare(onnxruntime URL ${ORT_URL} URL_HASH SHA256=${ORT_SHA256})
else()
    FetchContent_Declare(onnxruntime URL ${ORT_URL})
endif()

FetchContent_MakeAvailable(onnxruntime)

set(ONNXRUNTIME_ROOT "${onnxruntime_SOURCE_DIR}")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include" CACHE PATH "" FORCE)

# The release tgz ships libonnxruntime.<version>.dylib and a versionless
# symlink libonnxruntime.dylib. We link against the symlink and copy the real
# dylib into the .clap bundle's Frameworks/.
set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib" CACHE FILEPATH "" FORCE)
set(ONNXRUNTIME_DYLIB_REAL "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.${ORT_VERSION}.dylib" CACHE FILEPATH "" FORCE)
