cmake_minimum_required(VERSION 3.8)
project(BitonicSort)

set(CMAKE_CXX_STANDARD 11)

link_libraries(/usr/lib/x86_64-linux-gnu/libOpenCL.so)

include_directories(../shared)
link_directories(../shared)

set(SOURCE_FILES
        main.cpp
        BitonicSort.cpp
        BitonicSort.h
        ../shared/OpenCLMgr.cpp
        ../shared/OpenCLMgr.h
        ../shared/ansi_colors.h
        ../shared/clstatushelper.h
        )

add_executable(BitonicSort ${SOURCE_FILES})