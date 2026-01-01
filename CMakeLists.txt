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

# Find required packages
find_package(Vulkan REQUIRED)

# Find glslangValidator for shader compilation
find_program(GLSLANG_VALIDATOR glslangValidator
    HINTS $ENV{VULKAN_SDK}/bin
    REQUIRED
)

add_executable(gpusssp
    src/gpusssp.cpp
)

target_link_libraries(gpusssp
    PRIVATE
        Vulkan::Vulkan
)

# Custom command to compile GLSL shader to SPIR-V
set(SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/delta_step.comp)
set(SHADER_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/delta_step.spv)

add_custom_command(
    OUTPUT ${SHADER_OUTPUT}
    COMMAND ${GLSLANG_VALIDATOR} -V ${SHADER_SOURCE} -o ${SHADER_OUTPUT}
    DEPENDS ${SHADER_SOURCE}
    COMMENT "Compiling compute shader to SPIR-V"
    VERBATIM
)

add_custom_target(shaders
    DEPENDS ${SHADER_OUTPUT}
)
add_dependencies(gpusssp shaders)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(gpusssp PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

message(STATUS "CMake version: ${CMAKE_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Vulkan found: ${Vulkan_FOUND}")
message(STATUS "glslangValidator: ${GLSLANG_VALIDATOR}")
