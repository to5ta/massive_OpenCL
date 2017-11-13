cmake_minimum_required(VERSION 3.8)
project(HelloWorldCL)

link_libraries(/usr/lib/x86_64-linux-gnu/libOpenCL.so)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        HelloWorld.cpp
        HelloWorld_Kernel.cl)

add_executable(HelloWorldCL ${SOURCE_FILES})