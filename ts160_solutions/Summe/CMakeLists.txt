cmake_minimum_required(VERSION 3.8)
project(Summe)

set(CMAKE_CXX_STANDARD 11)

link_libraries(/usr/lib/x86_64-linux-gnu/libOpenCL.so)

set(SOURCE_FILES
        OpenCLMgr.cpp
        OpenCLMgr.h
        Summe.cpp)

add_executable(Summe ${SOURCE_FILES})