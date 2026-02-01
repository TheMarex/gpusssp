cmake_minimum_required(VERSION 3.20)
project(gpusssp
    VERSION 1.0.0
    DESCRIPTION "GPU SSSP implementation using Vulkan compute shaders"
    LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libosmium/include
)

# Find required packages
find_package(Vulkan REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)
find_package(BZip2 REQUIRED)

# Find glslangValidator for shader compilation
find_program(GLSLANG_VALIDATOR glslangValidator
    HINTS $ENV{VULKAN_SDK}/bin
    REQUIRED
)

# GPU SSSP executable
add_executable(gpusssp
    src/gpusssp.cpp
)

target_link_libraries(gpusssp
    PRIVATE
        Vulkan::Vulkan
)

# OSM to Graph converter executable
add_executable(osm2graph
    src/osm2graph.cpp
)

target_link_libraries(osm2graph
    PRIVATE
        ZLIB::ZLIB
        BZip2::BZip2
        Threads::Threads
)

# Generate queries executable
add_executable(generate_queries
    experiments/generate_queries.cpp
)

# Experiment Dijkstra
add_executable(experiment_dijkstra
    experiments/experiment_dijkstra.cpp
)

# Experiment DeltaStep
add_executable(experiment_deltastep
    experiments/experiment_deltastep.cpp
)

target_link_libraries(experiment_deltastep
    PRIVATE
        Vulkan::Vulkan
)

# Experiment BellmanFord
add_executable(experiment_bellmanford
    experiments/experiment_bellmanford.cpp
)

target_link_libraries(experiment_bellmanford
    PRIVATE
        Vulkan::Vulkan
)

# Experiment NearFar
add_executable(experiment_nearfar
    experiments/experiment_nearfar.cpp
)

target_link_libraries(experiment_nearfar
    PRIVATE
        Vulkan::Vulkan
)

# Custom command to compile GLSL shaders to SPIR-V
set(DELTA_STEP_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/delta_step.comp)
set(DELTA_STEP_SHADER_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/delta_step.spv)

add_custom_command(
    OUTPUT ${DELTA_STEP_SHADER_OUTPUT}
    COMMAND ${GLSLANG_VALIDATOR} -V ${DELTA_STEP_SHADER_SOURCE} --target-env vulkan1.1 -o ${DELTA_STEP_SHADER_OUTPUT}
    DEPENDS ${DELTA_STEP_SHADER_SOURCE}
    COMMENT "Compiling delta_step compute shader to SPIR-V"
    VERBATIM
)

set(BELLMAN_FORD_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/bellman_ford.comp)
set(BELLMAN_FORD_SHADER_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bellman_ford.spv)

add_custom_command(
    OUTPUT ${BELLMAN_FORD_SHADER_OUTPUT}
    COMMAND ${GLSLANG_VALIDATOR} -V ${BELLMAN_FORD_SHADER_SOURCE} --target-env vulkan1.1 -o ${BELLMAN_FORD_SHADER_OUTPUT}
    DEPENDS ${BELLMAN_FORD_SHADER_SOURCE}
    COMMENT "Compiling bellman_ford compute shader to SPIR-V"
    VERBATIM
)

set(NEARFAR_RELAX_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/nearfar_relax.comp)
set(NEARFAR_RELAX_SHADER_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/nearfar_relax.spv)

add_custom_command(
    OUTPUT ${NEARFAR_RELAX_SHADER_OUTPUT}
    COMMAND ${GLSLANG_VALIDATOR} -V ${NEARFAR_RELAX_SHADER_SOURCE} --target-env vulkan1.1 -o ${NEARFAR_RELAX_SHADER_OUTPUT}
    DEPENDS ${NEARFAR_RELAX_SHADER_SOURCE}
    COMMENT "Compiling nearfar_relax compute shader to SPIR-V"
    VERBATIM
)

set(NEARFAR_COMPACT_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/nearfar_compact.comp)
set(NEARFAR_COMPACT_SHADER_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/nearfar_compact.spv)

add_custom_command(
    OUTPUT ${NEARFAR_COMPACT_SHADER_OUTPUT}
    COMMAND ${GLSLANG_VALIDATOR} -V ${NEARFAR_COMPACT_SHADER_SOURCE} --target-env vulkan1.1 -o ${NEARFAR_COMPACT_SHADER_OUTPUT}
    DEPENDS ${NEARFAR_COMPACT_SHADER_SOURCE}
    COMMENT "Compiling nearfar_compact compute shader to SPIR-V"
    VERBATIM
)

set(NEARFAR_PREPARE_DISPATCH_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/nearfar_prepare_dispatch.comp)
set(NEARFAR_PREPARE_DISPATCH_SHADER_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/nearfar_prepare_dispatch.spv)

add_custom_command(
    OUTPUT ${NEARFAR_PREPARE_DISPATCH_SHADER_OUTPUT}
    COMMAND ${GLSLANG_VALIDATOR} -V ${NEARFAR_PREPARE_DISPATCH_SHADER_SOURCE} --target-env vulkan1.1 -o ${NEARFAR_PREPARE_DISPATCH_SHADER_OUTPUT}
    DEPENDS ${NEARFAR_PREPARE_DISPATCH_SHADER_SOURCE}
    COMMENT "Compiling nearfar_prepare_dispatch compute shader to SPIR-V"
    VERBATIM
)

add_custom_target(shaders
    DEPENDS ${DELTA_STEP_SHADER_OUTPUT} ${BELLMAN_FORD_SHADER_OUTPUT} ${NEARFAR_RELAX_SHADER_OUTPUT} ${NEARFAR_COMPACT_SHADER_OUTPUT} ${NEARFAR_PREPARE_DISPATCH_SHADER_OUTPUT}
)
add_dependencies(gpusssp shaders)
add_dependencies(experiment_deltastep shaders)
add_dependencies(experiment_bellmanford shaders)
add_dependencies(experiment_nearfar shaders)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# AddressSanitizer support
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

if(ENABLE_ASAN)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer -g")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ASAN_FLAGS}")
        message(STATUS "AddressSanitizer enabled")
    else()
        message(WARNING "AddressSanitizer requested but not supported with ${CMAKE_CXX_COMPILER_ID}")
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(gpusssp PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
    target_compile_options(osm2graph PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

# Testing support
option(BUILD_TESTS "Build tests" ON)

if(BUILD_TESTS)
    enable_testing()
    
    # Fetch Catch2
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.2
    )
    FetchContent_MakeAvailable(Catch2)
    
    # Add test executable
    add_executable(gpusssp_tests
        tests/test_main.cpp
        tests/test_deltastep.cpp
        tests/test_nearfar.cpp
        tests/test_string_util.cpp
        tests/test_graph_metrics.cpp
    )
    
    target_link_libraries(gpusssp_tests
        PRIVATE
            Catch2::Catch2WithMain
            Vulkan::Vulkan
    )
    
    # Make sure tests can find the shader
    add_dependencies(gpusssp_tests shaders)
    
    # Compiler warnings for tests
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(gpusssp_tests PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()
    
    # Discover tests
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(CTest)
    include(Catch)
    catch_discover_tests(gpusssp_tests)
    
    message(STATUS "Tests enabled")
endif()

# clang-format integration
find_program(CLANG_FORMAT_EXECUTABLE
    NAMES clang-format
    DOC "Path to clang-format executable"
)

if(CLANG_FORMAT_EXECUTABLE)
    # Collect all source files
    file(GLOB_RECURSE ALL_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.hpp
    )
    
    # Add format target
    add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXECUTABLE}
        -i
        -style=file
        ${ALL_SOURCE_FILES}
        COMMENT "Running clang-format on all source files"
        VERBATIM
    )
    
    # Add format-check target (useful for CI)
    add_custom_target(format-check
        COMMAND ${CLANG_FORMAT_EXECUTABLE}
        --dry-run
        --Werror
        -style=file
        ${ALL_SOURCE_FILES}
        COMMENT "Checking code formatting with clang-format"
        VERBATIM
    )
    
    message(STATUS "clang-format: ${CLANG_FORMAT_EXECUTABLE}")
    message(STATUS "  Run 'make format' or 'cmake --build . --target format' to format code")
    message(STATUS "  Run 'make format-check' or 'cmake --build . --target format-check' to check formatting")
else()
    message(STATUS "clang-format: NOT FOUND (formatting targets will not be available)")
endif()

message(STATUS "CMake version: ${CMAKE_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Vulkan found: ${Vulkan_FOUND}")
message(STATUS "glslangValidator: ${GLSLANG_VALIDATOR}")
message(STATUS "AddressSanitizer: ${ENABLE_ASAN}")
