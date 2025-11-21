# collect C++ and Objective-C files
file(GLOB_RECURSE AVK_MACOS_LAUNCHER_FILES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.m"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.mm"
)

add_executable(avk-macos-launcher MACOSX_BUNDLE)
target_compile_definitions(avk-macos-launcher
  PRIVATE
  "AVK_OS_${AVK_OS}" "AVK_COMPILER_${AVK_COMPILER}" "AVK_ARCH_${AVK_ARCH}")
target_compile_options(avk-macos-launcher PRIVATE ${AVK_CXX_TARGET_COMPILE_FLAGS})
target_link_libraries(avk-macos-launcher PRIVATE "-framework Cocoa" "-framework QuartzCore" avk::core)
target_sources(avk-macos-launcher
  PRIVATE
  ${AVK_MACOS_LAUNCHER_FILES}
)

set_target_properties(avk-macos-launcher PROPERTIES OUTPUT_NAME "avk-gui")