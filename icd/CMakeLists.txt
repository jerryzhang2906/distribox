cmake_minimum_required(VERSION 3.16)
project(distribox_icd VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Paths ───────────────────────────────────────────────
set(DISTRIBOX_THIRD_PARTY ${CMAKE_SOURCE_DIR}/../third_party/include)

# ── ICD library ─────────────────────────────────────────
add_library(distribox_icd SHARED
    icd.c
    api/platform.c
    api/device.c
    api/context.c
    api/commandqueue.c
    api/memory.c
    api/program.c
    api/kernel.c
    api/event.c
    ipc/ipc_client.c
    ipc/shm.c
)

target_include_directories(distribox_icd PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${DISTRIBOX_THIRD_PARTY}
)

# On Windows: also need to export the ICD entry point
if(WIN32)
    target_link_libraries(distribox_icd PRIVATE ws2_32)
    set_target_properties(distribox_icd PROPERTIES
        PREFIX ""  # No "lib" prefix on Windows
    )
endif()

# On macOS: force .dylib (not .so)
if(APPLE)
    set_target_properties(distribox_icd PROPERTIES
        SUFFIX ".dylib"
    )
endif()

# ── Tests ───────────────────────────────────────────────
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/icd_test.c)
    enable_testing()
    add_executable(icd_test test/icd_test.c)
    target_link_libraries(icd_test distribox_icd)
    target_include_directories(icd_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME icd_test COMMAND icd_test)
endif()
