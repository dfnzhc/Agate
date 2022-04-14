cmake_minimum_required(VERSION 3.21)

project(Agate VERSION 0.1 LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/cmake")

include(nvcuda_compile_ptx)
include(Macros)
include(CompilerInfo)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
            "MinSizeRel" "RelWithDebInfo")
endif ()

if (MSVC)
    list(APPEND AGATE_DEFINITIONS "_CRT_SECURE_NO_WARNINGS")
    list(APPEND AGATE_DEFINITIONS "_ENABLE_EXTENDED_ALIGNED_STORAGE")
endif ()

if (WIN32)
    list(APPEND AGATE_DEFINITIONS "AGATE_IN_WINDOWS" "NOMINMAX" "WIN32_LEAN_AND_MEAN")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif (WIN32)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# cuda
# Turn off the warning that NVCC issues when generating PTX from our CUDA samples.  This
# is a custom extension to the FindCUDA code distributed by CMake.
OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)

# CUDA 8 is broken for generating dependencies during configure
option(CUDA_GENERATE_DEPENDENCIES_DURING_CONFIGURE "Generate dependencies during configure time instead of only during build time." OFF)
# Find at least a 11.0 version of CUDA.
find_package(CUDA 11.0 REQUIRED)
# Present the CUDA_64_BIT_DEVICE_CODE on the default set of options.
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

# optix
# Locate the NVRT distribution.
set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0/" CACHE PATH "Path to OptiX installed location.")
# Search for the OptiX libraries and include files.
find_package(OptiX REQUIRED)

# Other dependencies
find_package(OpenGL REQUIRED)
add_subdirectory(ext)

include_directories(
        include
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OptiX_INCLUDE}
        ${CUDA_INCLUDE_DIRS}
)

set(HEADERS
        include/Agate/Agate.h
        include/Agate/Core/Log.h include/pch.h
        include/Agate/Core/Window.h
        include/Agate/Core/Common.h
        include/Agate/Util/Timer.h
        include/Agate/Core/Error.h
        include/Agate/Core/Aplication.h
        include/Agate/Core/OptixRenderer.h
        include/Agate/Util/ReadFile.h include/Agate/Util/CudaBuffer.h)


set(SOURCES
        src/Agate.cpp
        src/main.cpp
        src/pch.cpp
        src/Core/Window.cpp
        src/Core/Common.cpp
        src/Core/Aplication.cpp
        src/Core/OptixRenderer.cpp
        )


set(SHADERS
        include/Agate/Shader/optixHello.cu
        )


set(SHADERS_HEADERS
        include/Agate/Shader/vector_math.h 
        include/Agate/Shader/VertexInfo.h 
        include/Agate/Shader/LaunchParameter.h)

set_source_files_properties(
        ${SHADERS}

        PROPERTIES 
        LANGUAGE CUDA
#        HEADER_FILE_ONLY TRUE
        COMPILE_OPTIONS "--ptx"
        INCLUDE_DIRECTORIES ${OptiX_INCLUDE}
)

# just for syntax highlighting in Clion
add_library(ptx OBJECT ${SHADERS} ${SHADERS_HEADERS})
set_property(TARGET ptx PROPERTY CUDA_PTX_COMPILATION ON)

# compile .cu file to .ptx
set(PTX_TARGET_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)
NVCUDA_COMPILE_PTX(SOURCES ${SHADERS}
        DEPENDENCIES ${SHADERS_HEADERS}
        TARGET_PATH "${PTX_TARGET_DIR}/ptx"
        GENERATED_FILES PTX_SOURCES
        NVCC_OPTIONS "--gpu-architecture=compute_50" "--use_fast_math" "--relocatable-device-code=true" "-line-info" "-Wno-deprecated-gpu-targets" "-I${OptiX_INCLUDE}" "-I${CMAKE_CURRENT_SOURCE_DIR}/include/Agata/Shader"
        )

source_group("headers" FILES ${HEADERS})
source_group("sources" FILES ${SOURCES})
source_group("shaders" FILES ${SHADERS})
source_group("shaders_headers" FILES ${SHADERS_HEADERS})
source_group("ptx" FILES ${PTX_SOURCES})

add_executable(Agate
        ${HEADERS}
        ${SOURCES}
#        ${SHADERS}
        ${SHADERS_HEADERS}
        ${PTX_SOURCES}
        )

target_compile_definitions(Agate PRIVATE ${AGATE_DEFINITIONS})
target_precompile_headers(Agate PRIVATE include/pch.h)
#set_property(TARGET Agate PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#set_property(TARGET Agate PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(Agate
        ${CUDA_LIBRARIES}
        ${OPTIX_LIBRARIES}
        OpenGL::GL
        libglfw
        spdlog
        glew
        )

target_include_directories(Agate
        PUBLIC
        include
        ${GLEW_INCLUDE_DIRS}
        ${STB_INCLUDE_DIR}
        )
