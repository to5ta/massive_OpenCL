cmake_minimum_required(VERSION 3.8)
project(PrefixSum)

set(CMAKE_CXX_STANDARD 11)

link_libraries(/usr/lib/x86_64-linux-gnu/libOpenCL.so)

set(SOURCE_FILES    main.cpp
        ../shared/ansi_colors.h
        ../shared/clstatushelper.h
        ../shared/OpenCLMgr.h
        ../shared/OpenCLMgr.cpp
        ../shared/ansi_colors.h
        PrefixSum.cpp

        PrefixSum.h)

add_executable(PrefixSum ${SOURCE_FILES})


