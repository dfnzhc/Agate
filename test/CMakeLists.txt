cmake_minimum_required(VERSION 3.21)
project(CGT_Test LANGUAGES CXX CUDA)

find_package(Catch2 CONFIG REQUIRED)

#add_executable(tests
#        test.cpp 
#        test_uString.cpp
#        )
#
#target_link_libraries(tests PRIVATE Catch2::Catch2 CGT)

# These examples use the standard separate compilation
set(SOURCES_IDIOMATIC_EXAMPLES
        test.cpp
        )

string(REPLACE ".cpp" "" BASENAMES_IDIOMATIC_EXAMPLES "${SOURCES_IDIOMATIC_EXAMPLES}")
set(TARGETS_IDIOMATIC_EXAMPLES ${BASENAMES_IDIOMATIC_EXAMPLES})


foreach (name ${TARGETS_IDIOMATIC_EXAMPLES})
    add_executable(${name} ${name}.cpp)
endforeach ()

set(ALL_EXAMPLE_TARGETS
        ${TARGETS_IDIOMATIC_EXAMPLES}
        )

foreach (name ${ALL_EXAMPLE_TARGETS})
    target_include_directories(${name} PRIVATE ${CGT_INCLUDE_DIR})
    target_link_libraries(${name} PRIVATE Catch2::Catch2 Agate)
    set_property(TARGET ${name} PROPERTY CXX_STANDARD 17)
endforeach ()