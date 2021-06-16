cmake_minimum_required(VERSION 3.10)
project(tf_c_inference)

find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow/lib)

set(CMAKE_CXX_STANDARD 17)
add_subdirectory(pybind11)
pybind11_add_module(tf_c_inference src/tf_c_inference.cpp src/Model.cpp src/Tensor.cpp)

target_include_directories(tf_c_inference PRIVATE ../../include $ENV{HOME}/libtensorflow/include)

target_link_libraries (tf_c_inference PRIVATE "${TENSORFLOW_LIB}")