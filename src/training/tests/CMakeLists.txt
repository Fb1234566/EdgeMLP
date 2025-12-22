cmake_minimum_required(VERSION 3.15)

enable_testing()


include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0)
FetchContent_MakeAvailable(googletest)

file(GLOB TEST_SOURCES "*.cpp")

add_executable(tests ${TEST_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/Matrix.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/MLP.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/activation_functions/Sigmoid.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/activation_functions/Relu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/activation_functions/Tanh.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/activation_functions/Linear.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/Activation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/loss_functions/MSE.cpp
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
