cmake_minimum_required(VERSION 3.8)
project(MatrixAdd)

set(CMAKE_CXX_STANDARD 11)

link_libraries(/usr/lib/x86_64-linux-gnu/libOpenCL.so)

set(SOURCE_FILES
        Matrix.cpp
        Matrix.h
        MatrixMain.cpp
        OpenCLMgr.cpp
        OpenCLMgr.h)

add_executable(MatrixAdd ${SOURCE_FILES})