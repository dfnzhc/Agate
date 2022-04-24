# glfw
option(GLFW_BUILD_DOCS OFF)
option(GLFW_BUILD_TESTS OFF)
option(GLFW_BUILD_EXAMPLES OFF)
option(GLFW_BUILD_TESTS OFF)
option(GLFW_INSTALL OFF)
find_package(glfw3 REQUIRED CONFIG)
add_library(libglfw INTERFACE)
target_link_libraries(libglfw INTERFACE glfw)

# spdlog
find_package(spdlog REQUIRED CONFIG)
add_library(spdlog INTERFACE)
target_link_libraries(spdlog INTERFACE spdlog::spdlog spdlog::spdlog_header_only)

# stb
set(STB_INCLUDE_DIR stb)

# OpenGL and glad
add_library(glad SHARED
        KHR/khrplatform.h
        glad/glad.c
        glad/glad.h
        )
target_compile_definitions( glad
        PRIVATE GLAD_GLAPI_EXPORT_BUILD
        PUBLIC  GLAD_GLAPI_EXPORT )
target_include_directories( glad PUBLIC . )
target_link_libraries( glad PUBLIC ${OPENGL_LIBRARIES})
set_property(TARGET glad PROPERTY FOLDER Dependencies)

# tinygltf
set(TinyGLTF_INCLUDE_DIR tinygltf)

# tinyobj
set(TinyOBJ_INCLUDE_DIR tinyobjloader)
