cmake_minimum_required (VERSION 3.0)
cmake_policy(SET CMP0048 NEW)

project (annealing-lowlevel VERSION 0.1)
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/Modules")

find_package (Vulkan REQUIRED)
find_package (FFTW3 REQUIRED)

add_subdirectory (src)
add_subdirectory (shaders)
