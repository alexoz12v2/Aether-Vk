add_executable(avk-test-fibers)
target_compile_definitions(avk-test-fibers
  PRIVATE 
    "AVK_OS_${AVK_OS}" "AVK_COMPILER_${AVK_COMPILER}" "AVK_ARCH_${AVK_ARCH}" "AVK_${AVK_BUILD_FRAGMENT}"
    "UNICODE" "WIN32_LEAN_AND_MEAN" "NOMINMAX")
target_compile_options(avk-test-fibers PRIVATE ${AVK_CXX_TARGET_COMPILE_FLAGS})
target_sources(avk-test-fibers 
  PUBLIC
    FILE_SET avk_text_fibers_headers TYPE HEADERS BASE_DIRS public/ FILES
      public/fiber/mpmc.h
  PRIVATE 
    main.cpp 
)

if((AVK_OS STREQUAL "WINDOWS") AND (CMAKE_BUILD_TYPE STREQUAL "Debug") AND ${AVK_USE_SANITIZERS})
  target_link_libraries(avk-test-fibers PRIVATE Boost::fiber win-asan-dynamic)
else()
  target_link_libraries(avk-test-fibers PRIVATE Boost::fiber)
endif()

if(AVK_OS STREQUAL "WINDOWS")
  add_custom_command(TARGET avk-test-fibers POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:avk-test-fibers> $<TARGET_FILE_DIR:avk-test-fibers>
    COMMAND_EXPAND_LISTS
  )
endif()

install(TARGETS avk-test-fibers FILE_SET avk_text_fibers_headers)

add_library(avk-core)
target_compile_definitions(avk-core 
  PRIVATE 
    "AVK_OS_${AVK_OS}" "AVK_COMPILER_${AVK_COMPILER}" "AVK_ARCH_${AVK_ARCH}" "AVK_${AVK_BUILD_FRAGMENT}"
    "UNICODE" "WIN32_LEAN_AND_MEAN" "NOMINMAX")
target_compile_options(avk-core PUBLIC ${AVK_CXX_TARGET_COMPILE_FLAGS})
target_link_libraries(avk-core PUBLIC Boost::fiber Vulkan::Vulkan)
target_sources(avk-core
  PUBLIC FILE_SET avk_core_headers TYPE HEADERS BASE_DIRS public/ FILES
    public/fiber/mpmc.h public/fiber/jobs.h 
    public/render/context-vk.h public/render/pipeline-vk.h public/render/utils-vk.h
  PRIVATE
    private/fiber/jobs.cpp
    private/render/context-vk.cpp private/render/pipeline-vk.cpp private/render/utils-vk.cpp
)

target_include_directories(avk-core
  PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/public>
    $<INSTALL_INTERFACE:include>
)

add_library(avk::core ALIAS avk-core)
install(TARGETS avk-core FILE_SET avk_core_headers)
