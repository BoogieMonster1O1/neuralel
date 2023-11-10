cmake_minimum_required(VERSION 3.19)

project(neuralel CXX)

add_subdirectory(neuralnet)
include_directories(neuralnet/include)

add_subdirectory(isodd)
add_subdirectory(sine)
