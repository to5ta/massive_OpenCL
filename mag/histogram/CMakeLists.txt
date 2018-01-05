cmake_minimum_required(VERSION 3.8)
project(histogram)

set(CMAKE_CXX_STANDARD 11)

include_directories(include)
include_directories(src)

find_package(OpenCL)
if(OpenCL_FOUND)
    include_directories(${OPENL_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${OpenCL_LIBRARIES})
endif(OpenCL_FOUND)


find_package(Boost REQUIRED)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${Boost_LIBRARIES})
endif(Boost_FOUND)


set(SOURCE_FILES src/Histogram.cpp
        src/OpenCLMgr.cpp
        src/OpenCLMgr.h
        include/clstatushelper.h
        include/stb_image.h include/stb_image_write.h)


add_executable(histogram ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${LIBS})
