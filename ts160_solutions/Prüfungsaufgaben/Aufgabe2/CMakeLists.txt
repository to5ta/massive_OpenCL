cmake_minimum_required(VERSION 3.8)
project(Aufgabe2)

set(CMAKE_CXX_STANDARD 11)

link_libraries(/usr/lib/x86_64-linux-gnu/libOpenCL.so)

set(SOURCE_FILES    main.cpp
        ../shared/ansi_colors.h
        ../shared/clstatushelper.h
        ../shared/OpenCLMgr.cpp
        ../shared/OpenCLMgr.h
        PrefixSum.cpp
        PrefixSum.h)

add_executable(Aufgabe2 ${SOURCE_FILES})


