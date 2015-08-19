cmake_minimum_required(VERSION 3.2)
project(regsvd)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp SVD.cpp SVD.h IO.cpp IO.h Dataset.h)
add_executable(regsvd ${SOURCE_FILES})