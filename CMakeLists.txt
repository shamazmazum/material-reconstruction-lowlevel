cmake_minimum_required (VERSION 3.0)
cmake_policy(SET CMP0048 NEW)

project (voxcube VERSION 0.1)
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/Modules")

find_package (Vulkan REQUIRED)
find_package (FFTW3 REQUIRED)

include_directories (
  ${FFTW3_INCLUDE_DIR}
  ${Vulkan_INCLUDE_DIR}
)

add_library (annealing-lowlevel SHARED
  fft.c
  annealing-lowlevel.c)

target_link_libraries (annealing-lowlevel m ${FFTW3_LIBRARY} Vulkan::Vulkan)

install (TARGETS annealing-lowlevel LIBRARY DESTINATION lib)
