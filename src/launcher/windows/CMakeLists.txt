
add_executable(avk-windows-launcher WIN32)
target_compile_definitions(avk-windows-launcher
  PRIVATE 
    "AVK_OS_${AVK_OS}" "AVK_COMPILER_${AVK_COMPILER}" "AVK_ARCH_${AVK_ARCH}" "AVK_${AVK_BUILD_FRAGMENT}"
    "_UNICODE" "UNICODE" "WIN32_LEAN_AND_MEAN" "NOMINMAX")
target_compile_options(avk-windows-launcher PRIVATE ${AVK_CXX_TARGET_COMPILE_FLAGS})
target_sources(avk-windows-launcher 
  PRIVATE 
    main.cpp
)

if((AVK_OS STREQUAL "WINDOWS") AND (CMAKE_BUILD_TYPE STREQUAL "Debug") AND ${AVK_USE_SANITIZERS})
  target_link_libraries(avk-windows-launcher PRIVATE avk::core win-asan-dynamic)
else()
  target_link_libraries(avk-windows-launcher PRIVATE avk::core)
endif()

if(AVK_OS STREQUAL "WINDOWS")
  add_custom_command(TARGET avk-windows-launcher POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:avk-windows-launcher> $<TARGET_FILE_DIR:avk-windows-launcher>
    COMMAND_EXPAND_LISTS
  )
endif()

add_custom_target(avk-copy-static-shaders ALL 
  COMMAND ${CMAKE_COMMAND} -E copy ${avk_shader_file_list} ${CMAKE_CURRENT_BINARY_DIR}
)

add_dependencies(avk-windows-launcher avk-copy-static-shaders)