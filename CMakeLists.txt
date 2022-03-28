cmake_minimum_required(VERSION 3.21)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if (PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(
            FATAL_ERROR
            "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
    )
endif ()

project(CGT VERSION 0.1 LANGUAGES CXX CUDA)

add_subdirectory(ext)

set(CMAKE_CXX_STANDARD 23)
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
        ${STB_INCLUDE_DIR}
)

set(sources
        include/pch.h
        include/CGT/CGT.h
        src/common.cpp include/CGT/common.h
        src/util/uString.cpp include/CGT/util/uString.h
        src/util/uMath.cpp include/CGT/util/uMath.h

        src/util/timer.cpp include/CGT/util/timer.h src/util/uSampling.cpp include/CGT/util/uSampling.h src/misc/stb.cpp src/util/bitmap.cpp include/CGT/util/bitmap.h src/util/texture.cpp include/CGT/util/texture.h)

set_source_files_properties(${sources} PROPERTIES LANGUAGE CXX)

add_library(CGT
        # Source files
        ${sources}
        )

target_link_libraries(CGT
        PRIVATE
        tbb
        PUBLIC
        glm
        fmt)

target_precompile_headers(CGT PRIVATE include/pch.h)

set(CGT_INCLUDE_DIR PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(test)