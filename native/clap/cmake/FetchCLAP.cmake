# CLAP headers (MIT, header-only). Pin to a tagged release.
include(FetchContent)

FetchContent_Declare(clap
    GIT_REPOSITORY https://github.com/free-audio/clap.git
    GIT_TAG 1.2.2
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(clap)

set(CLAP_INCLUDE_DIR "${clap_SOURCE_DIR}/include" CACHE PATH "" FORCE)
