cmake_minimum_required(VERSION 3.25)
project(gpusssp
    VERSION 1.0.0
    DESCRIPTION "GPU SSSP implementation using Vulkan compute shaders"
    LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable BMI2 instruction set
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-mbmi2)
endif()

# Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libosmium/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
)

# Find required packages
find_package(Vulkan REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)
find_package(BZip2 REQUIRED)
find_package(glfw3 REQUIRED)

include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG v3.2
    SYSTEM
)
FetchContent_MakeAvailable(argparse)

# Find glslangValidator for shader compilation
find_program(GLSLANG_VALIDATOR glslangValidator
    HINTS $ENV{VULKAN_SDK}/bin
    REQUIRED
)

option(ENABLE_STATISTICS "Enable capturing algorithm statistics" OFF)

# GPU SSSP executable
add_executable(gpusssp
    src/gpusssp.cpp
    src/common/logger.cpp
)

target_link_libraries(gpusssp
    PRIVATE
        Vulkan::Vulkan
        argparse::argparse
)

if(ENABLE_STATISTICS)
    target_compile_definitions(gpusssp PRIVATE ENABLE_STATISTICS)
endif()

# OSM to Graph converter executable
add_executable(osm2graph
    src/osm2graph.cpp
    src/common/logger.cpp
)

target_link_libraries(osm2graph
    PRIVATE
        ZLIB::ZLIB
        BZip2::BZip2
        Threads::Threads
)

# Graph reorder executable
add_executable(graph_reorder
    src/graph_reorder.cpp
    src/common/logger.cpp
)

# Graph visualizer executable
add_executable(visualize_graph
    src/visualize_graph.cpp
    src/common/logger.cpp
)

target_link_libraries(visualize_graph
    PRIVATE
        Vulkan::Vulkan
        glfw
)

target_compile_definitions(visualize_graph PRIVATE ENABLE_TRACING)

# Generate queries executable
add_executable(generate_queries
    experiments/generate_queries.cpp
    src/common/logger.cpp
)

# Experiment Dijkstra
add_executable(experiment_dijkstra
    experiments/experiment_dijkstra.cpp
    src/common/logger.cpp
)

target_link_libraries(experiment_dijkstra
    PRIVATE
        argparse::argparse
)

if(ENABLE_STATISTICS)
    target_compile_definitions(experiment_dijkstra PRIVATE ENABLE_STATISTICS)
endif()

# Experiment Dial
add_executable(experiment_dial
    experiments/experiment_dial.cpp
    src/common/logger.cpp
)

target_link_libraries(experiment_dial
    PRIVATE
        argparse::argparse
)

if(ENABLE_STATISTICS)
    target_compile_definitions(experiment_dial PRIVATE ENABLE_STATISTICS)
endif()

# Experiment DeltaStep
add_executable(experiment_deltastep
    experiments/experiment_deltastep.cpp
    src/common/logger.cpp
)

target_link_libraries(experiment_deltastep
    PRIVATE
        Vulkan::Vulkan
        argparse::argparse
)

if(ENABLE_STATISTICS)
    target_compile_definitions(experiment_deltastep PRIVATE ENABLE_STATISTICS)
endif()

# Experiment BellmanFord
add_executable(experiment_bellmanford
    experiments/experiment_bellmanford.cpp
    src/common/logger.cpp
)

target_link_libraries(experiment_bellmanford
    PRIVATE
        Vulkan::Vulkan
        argparse::argparse
)

if(ENABLE_STATISTICS)
    target_compile_definitions(experiment_bellmanford PRIVATE ENABLE_STATISTICS)
endif()

# Experiment NearFar
add_executable(experiment_nearfar
    experiments/experiment_nearfar.cpp
    src/common/logger.cpp
)

target_link_libraries(experiment_nearfar
    PRIVATE
        Vulkan::Vulkan
        argparse::argparse
)

if(ENABLE_STATISTICS)
    target_compile_definitions(experiment_nearfar PRIVATE ENABLE_STATISTICS)
endif()

set(SHADER_COMPILE_FLAGS "")
if(ENABLE_STATISTICS)
    list(APPEND SHADER_COMPILE_FLAGS -DENABLE_STATISTICS)
endif()

set(SHADER_OUTPUTS "")
function(add_compute_shader SHADER_NAME)
    set(SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER_NAME}.comp)
    set(SHADER_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv)
    
    set(EXTRA_FLAGS ${ARGN})
    
    add_custom_command(
        OUTPUT ${SHADER_OUTPUT}
        COMMAND ${GLSLANG_VALIDATOR} -V ${SHADER_SOURCE} --target-env vulkan1.1 ${SHADER_COMPILE_FLAGS} ${EXTRA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/shaders -o ${SHADER_OUTPUT}
        DEPENDS ${SHADER_SOURCE}
        COMMENT "Compiling ${SHADER_NAME} compute shader to SPIR-V"
        VERBATIM
    )
    
    list(APPEND SHADER_OUTPUTS ${SHADER_OUTPUT})
    set(SHADER_OUTPUTS ${SHADER_OUTPUTS} PARENT_SCOPE)
    set(${SHADER_NAME}_SHADER_OUTPUT ${SHADER_OUTPUT} PARENT_SCOPE)
endfunction()

function(add_graphics_shader SHADER_NAME SHADER_STAGE)
    set(SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER_NAME}.${SHADER_STAGE})
    set(SHADER_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.${SHADER_STAGE}.spv)
    
    add_custom_command(
        OUTPUT ${SHADER_OUTPUT}
        COMMAND ${GLSLANG_VALIDATOR} -V ${SHADER_SOURCE} --target-env vulkan1.1 -o ${SHADER_OUTPUT}
        DEPENDS ${SHADER_SOURCE}
        COMMENT "Compiling ${SHADER_NAME}.${SHADER_STAGE} shader to SPIR-V"
        VERBATIM
    )
    
    list(APPEND SHADER_OUTPUTS ${SHADER_OUTPUT})
    set(SHADER_OUTPUTS ${SHADER_OUTPUTS} PARENT_SCOPE)
endfunction()

add_compute_shader(delta_step)
add_compute_shader(deltastep_prepare_dispatch)
add_compute_shader(bellman_ford)
add_compute_shader(nearfar_relax)
add_compute_shader(nearfar_compact)
add_compute_shader(nearfar_prepare_dispatch)
add_compute_shader(project_coordinates)
add_compute_shader(node_color)

add_graphics_shader(visualize_node vert)
add_graphics_shader(visualize_node frag)

add_custom_target(shaders DEPENDS ${SHADER_OUTPUTS})

add_dependencies(gpusssp shaders)
add_dependencies(experiment_deltastep shaders)
add_dependencies(experiment_bellmanford shaders)
add_dependencies(experiment_nearfar shaders)
add_dependencies(visualize_graph shaders)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;Profiling" CACHE STRING "" FORCE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo Profiling)

set(CMAKE_CXX_FLAGS_PROFILING "-Og -g3 -ggdb -fno-omit-frame-pointer -DNDEBUG" CACHE STRING
    "Flags for Profiling build type" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_PROFILING "" CACHE STRING
    "Linker flags for Profiling build type" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_PROFILING "" CACHE STRING
    "Shared linker flags for Profiling build type" FORCE)

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

include(CTest)
if(BUILD_TESTING)
    # Fetch Catch2
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.2
        SYSTEM
    )
    FetchContent_MakeAvailable(Catch2)
    
    # Add test executable
    add_executable(gpusssp_tests
        tests/test_main.cpp
        tests/test_deltastep.cpp
        tests/test_nearfar.cpp
        tests/test_string_util.cpp
        tests/test_graph_metrics.cpp
        tests/test_zorder.cpp
        tests/test_bucket_queue.cpp
        tests/test_circular_vector.cpp
        tests/test_dial.cpp
        src/common/logger.cpp
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
        ${CMAKE_CURRENT_SOURCE_DIR}/experiments/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/experiments/*.hpp
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

# clang-tidy integration
option(ENABLE_CLANG_TIDY "Enable clang-tidy static analysis" OFF)

find_program(CLANG_TIDY_EXECUTABLE
    NAMES clang-tidy
    DOC "Path to clang-tidy executable"
)

if(CLANG_TIDY_EXECUTABLE)
    if(ENABLE_CLANG_TIDY)
        # Enable clang-tidy for all targets during build
        set(CMAKE_CXX_CLANG_TIDY
            ${CLANG_TIDY_EXECUTABLE};
            --config-file=${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy;
        )
        message(STATUS "clang-tidy: ENABLED (will run automatically during build)")
    else()
        message(STATUS "clang-tidy: ${CLANG_TIDY_EXECUTABLE}")
    endif()
    
    message(STATUS "  Run 'cmake -DENABLE_CLANG_TIDY=ON ..' to enable clang-tidy during build")
else()
    message(STATUS "clang-tidy: NOT FOUND (static analysis targets will not be available)")
endif()

message(STATUS "CMake version: ${CMAKE_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Vulkan found: ${Vulkan_FOUND}")
message(STATUS "glslangValidator: ${GLSLANG_VALIDATOR}")
message(STATUS "AddressSanitizer: ${ENABLE_ASAN}")
