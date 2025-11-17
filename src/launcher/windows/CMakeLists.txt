# Automatically collect headers and sources
file(GLOB_RECURSE AVK_WINDOWS_LAUNCHER_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cxx"
)

# Exclude files in avk-winlib
list(FILTER AVK_WINDOWS_LAUNCHER_SOURCES EXCLUDE REGEX ".*/avk-winlib/.*")

# WinRT C++20 wrapper
add_subdirectory(avk-winlib)

add_executable(avk-windows-launcher WIN32)
target_compile_definitions(avk-windows-launcher
  PRIVATE 
    "AVK_OS_${AVK_OS}" "AVK_COMPILER_${AVK_COMPILER}" "AVK_ARCH_${AVK_ARCH}" "AVK_${AVK_BUILD_FRAGMENT}"
    "_UNICODE" "UNICODE" "WIN32_LEAN_AND_MEAN" "NOMINMAX")
target_compile_options(avk-windows-launcher PRIVATE ${AVK_CXX_TARGET_COMPILE_FLAGS})
target_sources(avk-windows-launcher 
  PRIVATE 
    ${AVK_WINDOWS_LAUNCHER_SOURCES}
)
target_include_directories(avk-windows-launcher PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(avk-windows-launcher PROPERTIES 
  OUTPUT_NAME "avk-gui"
  SUFFIX ".exe"
)

if((AVK_OS STREQUAL "WINDOWS") AND (CMAKE_BUILD_TYPE STREQUAL "Debug") AND ${AVK_USE_SANITIZERS})
  target_link_libraries(avk-windows-launcher PRIVATE avk::win-comutils avk::core win-asan-dynamic dwmapi)
else()
  target_link_libraries(avk-windows-launcher PRIVATE avk::win-comutils avk::core dwmapi)
endif()

if(AVK_OS STREQUAL "WINDOWS")
  add_custom_command(TARGET avk-windows-launcher POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:avk-windows-launcher> $<TARGET_FILE_DIR:avk-windows-launcher>
    COMMAND_EXPAND_LISTS
  )
endif()

add_custom_target(avk-copy-static-shaders ALL
  COMMAND ${CMAKE_COMMAND} -E copy ${avk_shader_file_list} ${CMAKE_CURRENT_BINARY_DIR}
)

# TODO better
add_custom_target(avk-copy-assets ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/assets
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets ${CMAKE_CURRENT_BINARY_DIR}/assets
)

add_dependencies(avk-windows-launcher avk-copy-static-shaders avk-copy-assets)