cmake_minimum_required(VERSION 3.8)
project(Aufgabe1)

set(CMAKE_CXX_STANDARD 11)

link_libraries(/usr/lib/x86_64-linux-gnu/libOpenCL.so)

#include_directories(../shared)
#link_directories(../shared)
#include_directories(.)

set(SOURCE_FILES
        main.cpp
        ../shared/OpenCLMgr.cpp
        ../shared/OpenCLMgr.h
        ../shared/clstatushelper.h
        ../shared/ansi_colors.h
        ../shared/stb_image.h
        Histogram.cpp
        Histogram.h
        )

add_executable(Aufgabe1 ${SOURCE_FILES})