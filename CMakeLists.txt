cmake_minimum_required (VERSION 3.0)
cmake_policy(SET CMP0048 NEW)

project (voxcube VERSION 0.1)
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/Modules")

find_package (Vulkan REQUIRED)
find_package (FFTW3 REQUIRED)

include(CompileShader)
compile_shader(update-shader
  SOURCE ${CMAKE_SOURCE_DIR}/shaders/update-s2.comp
  TARGET ${CMAKE_BINARY_DIR}/shaders/update-s2.spv
)

compile_shader(metric-shader
  SOURCE ${CMAKE_SOURCE_DIR}/shaders/metric.comp
  TARGET ${CMAKE_BINARY_DIR}/shaders/metric.spv
)

compile_shader(reduce-shader
  SOURCE ${CMAKE_SOURCE_DIR}/shaders/reduce.comp
  TARGET ${CMAKE_BINARY_DIR}/shaders/reduce.spv
)

configure_file(${CMAKE_SOURCE_DIR}/shader-source.h.in ${CMAKE_BINARY_DIR}/shader-source.h)

add_library (annealing-lowlevel SHARED
  fft.c
  annealing-lowlevel.c)

target_link_libraries (annealing-lowlevel m ${FFTW3_LIBRARY} Vulkan::Vulkan)
target_include_directories(annealing-lowlevel PUBLIC
  ${FFTW3_INCLUDE_DIR}
  ${Vulkan_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}
)
add_dependencies(annealing-lowlevel update-shader reduce-shader metric-shader)

install (TARGETS annealing-lowlevel LIBRARY DESTINATION lib)
install (DIRECTORY ${CMAKE_BINARY_DIR}/shaders DESTINATION share/annealing-lowlevel)
