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

# Automatically collect headers and sources
file(GLOB_RECURSE AVK_CORE_PUBLIC_HEADERS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/public/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/public/*.hpp"
)
file(GLOB_RECURSE AVK_CORE_PRIVATE_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/private/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/private/*.cxx"
)

add_library(avk-core)
target_compile_definitions(avk-core 
  PUBLIC "VK_NO_PROTOTYPES"
  PRIVATE 
    "AVK_OS_${AVK_OS}" "AVK_COMPILER_${AVK_COMPILER}" "AVK_ARCH_${AVK_ARCH}" "AVK_${AVK_BUILD_FRAGMENT}"
    "UNICODE" "WIN32_LEAN_AND_MEAN" "NOMINMAX")
target_compile_options(avk-core PUBLIC ${AVK_CXX_TARGET_COMPILE_FLAGS})
if(AVK_OS STREQUAL "MACOS")
  target_link_libraries(avk-core 
    PUBLIC Boost::fiber Vulkan::Headers GPUOpen::VulkanMemoryAllocator volk::volk volk::volk_headers Vulkan::MoltenVK)
else()
  target_link_libraries(avk-core 
    PUBLIC Boost::fiber Vulkan::Headers GPUOpen::VulkanMemoryAllocator volk::volk volk::volk_headers)
endif()
target_sources(avk-core
  PUBLIC FILE_SET avk_core_headers TYPE HEADERS BASE_DIRS public/ FILES
    ${AVK_CORE_PUBLIC_HEADERS}
  PRIVATE
    ${AVK_CORE_PRIVATE_SOURCES}
)

target_include_directories(avk-core
  PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/public>
    $<INSTALL_INTERFACE:include>
)

add_library(avk::core ALIAS avk-core)
install(TARGETS avk-core FILE_SET avk_core_headers)
