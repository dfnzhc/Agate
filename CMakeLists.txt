cmake_minimum_required(VERSION 3.21)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if (PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(
            FATAL_ERROR
            "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
    )
endif ()

project(CGT VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_CUDA "Enable CUDA" ON)

# Can not set cuda properly
if (NOT CMAKE_CUDA_COMPILER OR NOT ENABLE_CUDA)
    add_definitions(-DENABLE_CUDA)
endif ()

# Enable folders for projects in Visual Studio
if (CMAKE_GENERATOR MATCHES "Visual Studio")
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif ()

if (MSVC)
    add_compile_options(/MP)
    add_compile_options(/D_CRT_SECURE_NO_WARNINGS)

endif ()

include_directories(
        # CGT include files
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(headers
        include/pch.h
        include/CGT/CGT.h
        )

set_source_files_properties(${headers} PROPERTIES LANGUAGE CXX)

add_library(CGT
        # Header files
        ${headers}
        )

target_precompile_headers(CGT PRIVATE include/pch.h)


add_subdirectory(test)