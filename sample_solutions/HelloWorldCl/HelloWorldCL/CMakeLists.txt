cmake_minimum_required(VERSION 3.8)
project(HelloWorldCL)

link_libraries(/usr/lib/x86_64-linux-gnu/libOpenCL.so)

include_directories(/home/tosta/archive/studium/CS3_Master/Advanced_Programming_of_Massively_Parallel_Processors/HelloWorldCl/HelloWorldCL)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        HelloWorld.cpp
        HelloWorld_Kernel.cl)

add_executable(HelloWorldCL ${SOURCE_FILES})