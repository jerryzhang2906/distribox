cmake_minimum_required(VERSION 3.16)
project(distribox_engine VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Paths ───────────────────────────────────────────────
set(DISTRIBOX_THIRD_PARTY ${CMAKE_SOURCE_DIR}/../third_party/include)

# ── OpenCL detection ────────────────────────────────────
find_package(OpenCL QUIET)
if(OpenCL_FOUND)
    message(STATUS "OpenCL found: ${OpenCL_INCLUDE_DIRS}")
    add_definitions(-DHAS_OPENCL)
else()
    message(STATUS "OpenCL runtime not found - building CPU-only backend (OpenCL headers from third_party)")
endif()

# ── Engine library ──────────────────────────────────────
# engine_opencl.c contains both OpenCL and CPU fallback paths
add_library(distribox_engine SHARED
    src/engine_opencl.c
)

target_include_directories(distribox_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${DISTRIBOX_THIRD_PARTY}
)

if(OpenCL_FOUND)
    target_include_directories(distribox_engine PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(distribox_engine PRIVATE ${OpenCL_LIBRARIES})
endif()

# Platform-specific settings
if(WIN32)
    target_link_libraries(distribox_engine PRIVATE ws2_32)
    set_target_properties(distribox_engine PROPERTIES PREFIX "")
elseif(ANDROID)
    find_library(log-lib log)
    target_link_libraries(distribox_engine PRIVATE ${log-lib})
endif()

# ── Tests ───────────────────────────────────────────────
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/engine_test.c)
    enable_testing()
    add_executable(engine_test
        test/engine_test.c
    )
    target_link_libraries(engine_test distribox_engine)
    target_include_directories(engine_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME engine_test COMMAND engine_test)
endif()
