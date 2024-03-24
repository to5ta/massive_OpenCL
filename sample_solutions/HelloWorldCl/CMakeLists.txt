cmake_minimum_required(VERSION 3.8)
project(HelloWorldCl)

set(CMAKE_CXX_STANDARD 11)

link_libraries(/usr/lib/x86_64-linux-gnu/libOpenCL.so)

set(SOURCE_FILES
        HelloWorldCL/HelloWorld.cpp
        HelloWorldCL/HelloWorld_Kernel.cl)

add_executable(HelloWorldCl ${SOURCE_FILES})