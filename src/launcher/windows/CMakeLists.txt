
add_executable(avk-windows-launcher WIN32)
target_compile_definitions(avk-windows-launcher
  PRIVATE 
    "AVK_OS_${AVK_OS}" "AVK_COMPILER_${AVK_COMPILER}" "AVK_ARCH_${AVK_ARCH}"
    "UNICODE" "WIN32_LEAN_AND_MEAN" "NOMINMAX")
target_compile_options(avk-windows-launcher PRIVATE ${AVK_CXX_TARGET_COMPILE_FLAGS})
target_sources(avk-windows-launcher PRIVATE main.cpp)