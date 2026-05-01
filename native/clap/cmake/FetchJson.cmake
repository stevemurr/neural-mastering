# nlohmann/json: MIT, single-header. Used to parse plugin_meta.json.
include(FetchContent)

FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(json)

set(JSON_INCLUDE_DIR "${json_SOURCE_DIR}/single_include" CACHE PATH "" FORCE)
