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

# glew
find_package(GLEW REQUIRED CONFIG)
add_library(glew INTERFACE)
target_link_libraries(glew INTERFACE GLEW::GLEW)