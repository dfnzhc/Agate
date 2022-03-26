project(CGT_Test LANGUAGES CXX CUDA)

find_package(Catch2 CONFIG REQUIRED)
add_executable(tests test.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2 CGT)

## These examples use the standard separate compilation
#set( SOURCES_IDIOMATIC_EXAMPLES
#    030-Asn-Require-Check.cpp
#    100-Fix-Section.cpp
#    110-Fix-ClassFixture.cpp
#    120-Bdd-ScenarioGivenWhenThen.cpp
#    210-Evt-EventListeners.cpp
#    300-Gen-OwnGenerator.cpp
#    301-Gen-MapTypeConversion.cpp
#    302-Gen-Table.cpp
#    310-Gen-VariablesInGenerators.cpp
#    311-Gen-CustomCapture.cpp
#)
#
#string( REPLACE ".cpp" "" BASENAMES_IDIOMATIC_EXAMPLES "${SOURCES_IDIOMATIC_EXAMPLES}" )
#set( TARGETS_IDIOMATIC_EXAMPLES ${BASENAMES_IDIOMATIC_EXAMPLES} )
#
#
#foreach( name ${TARGETS_IDIOMATIC_EXAMPLES} )
#    add_executable( ${name}
#      ${EXAMPLES_DIR}/${name}.cpp )
#endforeach()
#
#set(ALL_EXAMPLE_TARGETS
#  ${TARGETS_IDIOMATIC_EXAMPLES}
#  010-TestCase
#  020-MultiFile
#)
#
#foreach( name ${ALL_EXAMPLE_TARGETS} )
#    target_link_libraries( ${name} Catch2 Catch2WithMain )
#    set_property(TARGET ${name} PROPERTY CXX_STANDARD 14)
#    set_property(TARGET ${name} PROPERTY CXX_EXTENSIONS OFF)
#endforeach()