# glm library
find_package(glm CONFIG REQUIRED)
add_library(glm INTERFACE)
target_link_libraries(glm INTERFACE glm::glm)

# fmt library
find_package(fmt CONFIG REQUIRED)
add_library(fmt INTERFACE)
target_link_libraries(fmt INTERFACE fmt::fmt-header-only)

# tbb
find_package(TBB CONFIG REQUIRED)
add_library(tbb INTERFACE)
target_link_libraries(tbb INTERFACE TBB::tbb)

# stb
set(STB_INCLUDE_DIR stb)