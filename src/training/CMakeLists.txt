cmake_minimum_required(VERSION 3.15)

project(EdgeMLP VERSION 0.1.0 LANGUAGES CXX)

# Standard C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SOURCES
        src/main.cpp
        include/Matrix.h
        src/Matrix.cpp
        src/activation_functions/Sigmoid.cpp
        src/Activation.cpp
        src/activation_functions/Relu.cpp
        src/activation_functions/Tanh.cpp
        src/activation_functions/Linear.cpp
        src/MLP.cpp
        src/loss_functions/MSE.cpp
)

set(OMP_NUM_THREADS 8)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)

option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/training)
    add_subdirectory(test)
endif()
