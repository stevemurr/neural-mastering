cmake_minimum_required(VERSION 3.22)

# The concrete plugin name comes from plugin_meta.json, not CMake. We build a
# single generic dylib; ``build.sh`` packages it as a per-model .clap bundle
# with the model-specific metadata.
project(nablafx_clap CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT APPLE)
    message(FATAL_ERROR "v1 targets macOS arm64 only. Cross/universal builds are v2.")
endif()

if(NOT CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)
endif()

include(cmake/FetchCLAP.cmake)
include(cmake/FetchOnnxRuntime.cmake)
include(cmake/FetchJson.cmake)

add_library(nablafx_clap MODULE
    src/nablafx_plugin.cpp
    src/meta.cpp
    src/ort_session.cpp
    src/param_id.cpp
)

target_include_directories(nablafx_clap PRIVATE
    ${CLAP_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${JSON_INCLUDE_DIR}
)

target_link_libraries(nablafx_clap PRIVATE
    ${ONNXRUNTIME_LIBRARY}
    "-framework CoreFoundation"
)

# Composite TONE plugin: separate dylib that bundles auto-EQ + saturator +
# LA-2A behind a single AMT/TRM control surface, with the LUFS leveler and
# true-peak ceiling running in-host. spectral_mask_eq.hpp + rational_a.hpp
# are header-only so they don't appear in the source list.
add_library(tone_clap MODULE
    src/tone_plugin.cpp
    src/tone_gui.mm
    src/composite_meta.cpp
    src/meta.cpp
    src/lufs_leveler.cpp
    src/true_peak_ceiling.cpp
    src/param_id.cpp
)

set_source_files_properties(src/tone_gui.mm PROPERTIES
    COMPILE_FLAGS "-fobjc-arc"
)

target_include_directories(tone_clap PRIVATE
    ${CLAP_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${JSON_INCLUDE_DIR}
)

target_link_libraries(tone_clap PRIVATE
    ${ONNXRUNTIME_LIBRARY}
    "-framework CoreFoundation"
    "-framework Cocoa"
    "-framework WebKit"
    "-framework Accelerate"   # vDSP_fft for SpectralMaskEq
)

# Install the onnxruntime dylib next to ours and make each plugin find it
# via @loader_path/../Frameworks once it's inside its .clap bundle.
foreach(_t nablafx_clap tone_clap)
    set_target_properties(${_t} PROPERTIES
        PREFIX ""
        SUFFIX ".so"                 # build.sh renames to the bundle executable
        INSTALL_RPATH "@loader_path/../Frameworks"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endforeach()

# Expose artifacts to build.sh via a tiny config file
configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/build_config.cmake.in
    ${CMAKE_BINARY_DIR}/build_config.sh
    @ONLY
)
