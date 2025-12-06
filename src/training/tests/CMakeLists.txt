cmake_minimum_required(VERSION 3.15)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/52eb8108c5bdec04579160ae17225d66034bd723.zip
)
FetchContent_MakeAvailable(googletest)

file(GLOB TEST_SOURCES "*.cpp")

add_executable(tests ${TEST_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/Matrix.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/activation_functions/Sigmoid.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/activation_functions/Relu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/activation_functions/Tanh.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/activation_functions/Linear.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/Activation.cpp
)

target_include_directories(tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(tests
        gtest
        gtest_main
)

include(GoogleTest)

gtest_discover_tests(tests)
