configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/shader-source.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/shader-source.h
)

add_library (annealing-lowlevel SHARED
  fft.c
  annealing-lowlevel.c)

target_link_libraries (annealing-lowlevel m ${FFTW3_LIBRARY} Vulkan::Vulkan)
target_include_directories(annealing-lowlevel PUBLIC
  ${FFTW3_INCLUDE_DIR}
  ${Vulkan_INCLUDE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(annealing-lowlevel update-shader reduce-shader metric-shader)

install (TARGETS annealing-lowlevel LIBRARY DESTINATION lib)
