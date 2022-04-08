cmake_minimum_required(VERSION 3.21)

project(CGT VERSION 0.1 LANGUAGES CXX C CUDA)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/cmake")

include("nvcuda_compile_ptx")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
            "MinSizeRel" "RelWithDebInfo")
endif ()

# Configuration options
option(CGT_FLOAT_AS_DOUBLE "Use 64-bit floats" ON)
option(CGT_DBG_LOGGING "Enable debug logging" ON)
option(CGT_USE_GPU "Enable GPU functionality" ON)

if (MSVC)
    list(APPEND CGT_DEFINITIONS "_CRT_SECURE_NO_WARNINGS")
    list(APPEND CGT_DEFINITIONS "_ENABLE_EXTENDED_ALIGNED_STORAGE")
endif ()

if (CGT_FLOAT_AS_DOUBLE)
    list(APPEND CGT_DEFINITIONS "CGT_FLOAT_AS_DOUBLE")
endif ()

if (CGT_DBG_LOGGING)
    list(APPEND CGT_DEFINITIONS "CGT_DBG_LOGGING")
endif ()

if (WIN32)
    list(APPEND CGT_DEFINITIONS "CGT_IN_WINDOWS" "NOMINMAX" "WIN32_LEAN_AND_MEAN")
endif (WIN32)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

if (WIN32 AND "${CMAKE_GENERATOR}" MATCHES "^(Visual Studio).*")
    # Set the base folder where the per-project ptx folders get created.
    set(PTX_TARGET_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$(ConfigurationName)")
    # Enable multi-processor build on all Visual Studio versions.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else ()
    # DAR This should be independent of ${CMAKE_BUILD_TYPE} because that single-configuration generator will not create subfolders, will it?
    # Otherwise add something with if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(PTX_TARGET_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif ()

# optix
set(CGT_OPTIX7_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0")
find_path(OPTIX7_INCLUDE_DIR optix_7_host.h ${CGT_OPTIX7_PATH}/include)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX7 DEFAULT_MSG OPTIX7_INCLUDE_DIR)
mark_as_advanced(OPTIX7_INCLUDE_DIR)
set(OPTIX_INCLUDE_DIR "${OPTIX7_INCLUDE_DIR}")

# cuda
find_package(CUDAToolkit 11.0 REQUIRED)
if (CGT_USE_GPU)
    if (NOT ("${OPTIX_INCLUDE_DIR}" STREQUAL "") AND ${CUDAToolkit_FOUND})
        list(APPEND CGT_DEFINITIONS "CGT_USE_GPU")
    endif ()
endif ()

set(HEADERS
        include/CGT/CGT.h
        src/CGT.cpp src/util/Log.cpp include/CGT/util/Log.h)

#set(SOURCES
#        )

source_group("headers" FILES ${HEADERS})
source_group("sources" FILES ${SOURCES})


if (CGT_USE_GPU)
    set_source_files_properties(
            src/CGT.cpp src/util/Log.cpp
            
            PROPERTIES LANGUAGE CUDA
    )
endif ()

add_library(CGT STATIC
        ${HEADERS}
        #        ${SOURCES}
        )

if (CGT_USE_GPU)
    set_property(TARGET CGT PROPERTY CUDA_SEPARABLE_COMPILATION ON)

endif ()

target_compile_definitions(CGT PRIVATE ${CGT_DEFINITIONS})
set_property(TARGET CGT PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(CGT
        CUDA::cuda_driver
        )

target_include_directories(CGT
        PUBLIC
        include
        ${OPTIX_INCLUDE_DIR})

