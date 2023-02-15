cmake_minimum_required(VERSION 3.22)
project(PResize)
#Default target for module

if(NOT DEFINED ${PMERGE_TARGET})
    set(PMERGE_TARGET cmake-profile)
endif()
# set(PMERGE_TARGET cmake-profile)
# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/stb/)

add_halide_generator(resizeGenerator SOURCES generate_resize.cpp)

add_halide_library(resize FROM resizeGenerator
        TARGETS ${PMERGE_TARGET}
        AUTOSCHEDULER Halide::Mullapudi2016
        GENERATOR resize)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#add_executable(PResize main.cpp)
#target_link_libraries(PResize resize)
