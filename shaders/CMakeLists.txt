include(CompileShader)
compile_shader(update-shader
  SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/update-s2.comp
  TARGET ${CMAKE_CURRENT_BINARY_DIR}/update-s2.spv
)

compile_shader(metric-shader
  SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/metric.comp
  TARGET ${CMAKE_CURRENT_BINARY_DIR}/metric.spv
)

compile_shader(reduce-shader
  SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/reduce.comp
  TARGET ${CMAKE_CURRENT_BINARY_DIR}/reduce.spv
)

install (FILES
  ${CMAKE_CURRENT_BINARY_DIR}/reduce.spv
  ${CMAKE_CURRENT_BINARY_DIR}/metric.spv
  ${CMAKE_CURRENT_BINARY_DIR}/update-s2.spv
  DESTINATION share/annealing-lowlevel
)
