add_executable(avk-macos-launcher MACOSX_BUNDLE)
target_compile_definitions(avk-macos-launcher
  PRIVATE
    "AVK_OS_${AVK_OS}" "AVK_COMPILER_${AVK_COMPILER}" "AVK_ARCH_${AVK_ARCH}")
target_compile_options(avk-macos-launcher PRIVATE ${AVK_CXX_TARGET_COMPILE_FLAGS})
target_link_libraries(avk-macos-launcher PRIVATE "-framework Cocoa")
# target_link_libraries(avk-macos-launcher PRIVATE Vulkan::Vulkan)
target_sources(avk-macos-launcher
  PRIVATE
    main.mm
)
