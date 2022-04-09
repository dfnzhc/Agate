cmake_minimum_required(VERSION 3.21)

project(Agate VERSION 0.1 LANGUAGES CXX C CUDA)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/cmake")

include("nvcuda_compile_ptx")
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
set(AGATE_OPTIX7_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0")
find_path(OPTIX7_INCLUDE_DIR optix_7_host.h ${AGATE_OPTIX7_PATH}/include)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX7 DEFAULT_MSG OPTIX7_INCLUDE_DIR)
mark_as_advanced(OPTIX7_INCLUDE_DIR)
set(OPTIX_INCLUDE_DIR "${OPTIX7_INCLUDE_DIR}")

# cuda
find_package(CUDAToolkit 11.0 REQUIRED)


set(HEADERS
        include/Agate/Agate.h
        src/Agate.cpp)

include_directories(include)

#set(SOURCES
#        )

source_group("headers" FILES ${HEADERS})
source_group("sources" FILES ${SOURCES})


#    set_source_files_properties(
#            src/Agate.cpp src/util/Log.cpp
#            
#            PROPERTIES LANGUAGE CUDA
#    )

add_library(Agate STATIC
        ${HEADERS}
        #        ${SOURCES}
        )

target_compile_definitions(Agate PRIVATE ${AGATE_DEFINITIONS})
set_property(TARGET Agate PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET Agate PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(Agate
        CUDA::cuda_driver
        )

target_include_directories(Agate
        PUBLIC
        include
        ${OPTIX_INCLUDE_DIR})

