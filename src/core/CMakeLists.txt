add_executable(avk-test-fibers)
target_compile_definitions(avk-test-fibers
  PRIVATE 
    "AVK_OS_${AVK_OS}" "AVK_COMPILER_${AVK_COMPILER}" "AVK_ARCH_${AVK_ARCH}"
    "UNICODE" "WIN32_LEAN_AND_MEAN" "NOMINMAX")
target_compile_options(avk-test-fibers PRIVATE ${AVK_CXX_TARGET_COMPILE_FLAGS})
target_sources(avk-test-fibers PRIVATE main.cpp jump_fcontext-win-x86_64.S make_fcontext-win-x86_64.S)