cmake_minimum_required(VERSION 3.8)
project(ImageFX)

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")



#set(CMAKE_EXE_LINKER_FLAGS "sdl-config --cflags --libs")
set(CMAKE_EXE_LINKER_FLAGS "-v")

set(CMAKE_CXX_FLAGS "-Wno-c++11-narrowing")

find_package(OpenCL REQUIRED)
find_package(SDL REQUIRED)
find_package(SDL_ttf REQUIRED)
find_package(SDL_gfx REQUIRED)
find_package(SDL_image REQUIRED)

set(SOURCE_FILES
        ImageFX/OpenCLMgr.cpp
        ImageFX/OpenCLMgr.h
        ImageFX/stdafx.cpp
        ImageFX/stdafx.h
        ImageFX/ImageFX.cpp)

add_executable(ImageFX ${SOURCE_FILES})

target_link_libraries(ImageFX OpenCL::OpenCL)
target_link_libraries(ImageFX SDL)
target_link_libraries(ImageFX SDL_ttf)
target_link_libraries(ImageFX SDL_gfx)
target_link_libraries(ImageFX SDL_image)

