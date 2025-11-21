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
target_link_libraries(avk-macos-launcher PRIVATE "-framework Cocoa" "-framework QuartzCore" "-framework IOKit" avk::core)
target_sources(avk-macos-launcher
  PRIVATE
  ${AVK_MACOS_LAUNCHER_FILES}
)

# alternatively, you can use `set_source_files_properties(${shader_files} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")`
# avk-gui.app/
# ├─ Contents/
# │  ├─ MacOS/       <-- executable goes here
# │  ├─ Resources/   <-- assets should go here
# │  └─ Info.plist
add_custom_command(TARGET avk-macos-launcher POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_BUNDLE_CONTENT_DIR:avk-macos-launcher>/Resources/assets $<TARGET_BUNDLE_CONTENT_DIR:avk-macos-launcher>/Resources/shaders
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets $<TARGET_BUNDLE_CONTENT_DIR:avk-macos-launcher>/Resources/assets
  COMMAND ${CMAKE_COMMAND} -E copy ${avk_shader_file_list} $<TARGET_BUNDLE_CONTENT_DIR:avk-macos-launcher>/Resources/shaders
)

set_target_properties(avk-macos-launcher PROPERTIES OUTPUT_NAME "avk-gui")
add_dependencies(avk-macos-launcher avk-copy-assets avk-copy-static-shaders)

### Now for fixing up all our framework bundles and produce a self sufficient

# Tell the app to look in the Frameworks folder inside the bundle.
# Note: @executable_path is avk-gui.app/Contents/MacOS
#      -> ../Frameworks -> avk-gui.app/Contents/Frameworks
set_target_properties(avk-macos-launcher PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "@executable_path/../Frameworks"
)

## Now copy 3rd part frameworks inside the bundle (only MoltenVK)
set(MVK_FRAMEWORK_SRC "${MVK_FRAMEWORK}")

# post build: copy framework, fix rpaths, code-sign
# (you should do code sign manually, with "-" substituted with your apple developer certificate)
add_custom_command(TARGET avk-macos-launcher POST_BUILD
  COMMAND ${CMAKE_COMMAND}
  -DAPP_BUNDLE="$<TARGET_BUNDLE_DIR:avk-macos-launcher>"
  -DMVK_FRAMEWORK_SRC="${MVK_FRAMEWORK}"
  -P "${CMAKE_SOURCE_DIR}/cmake/FixupBundle.cmake"
)