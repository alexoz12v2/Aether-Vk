# Automatically collect headers and sources
file(GLOB_RECURSE AVK_CORE_PUBLIC_HEADERS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/public/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/public/*.hpp"
)
file(GLOB_RECURSE AVK_CORE_PRIVATE_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/private/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/private/*.cxx"
)

set(avk_core_libraries_public
  Boost::fiber Vulkan::Headers GPUOpen::VulkanMemoryAllocator volk::volk volk::volk_headers
)
set(avk_core_libraries_private "")
if(AVK_OS STREQUAL "MACOS")
  list(APPEND avk_core_libraries_public Vulkan::MoltenVK)
endif()
if(AVK_OS STREQUAL "WINDOWS")
  list(APPEND avk_core_libraries_private dwmapi)
endif()

add_library(avk-core)
target_compile_definitions(avk-core 
  PUBLIC "VK_NO_PROTOTYPES"
  PRIVATE 
    "AVK_OS_${AVK_OS}" "AVK_COMPILER_${AVK_COMPILER}" "AVK_ARCH_${AVK_ARCH}" "AVK_${AVK_BUILD_FRAGMENT}"
    "UNICODE" "WIN32_LEAN_AND_MEAN" "NOMINMAX")
target_compile_options(avk-core PUBLIC ${AVK_CXX_TARGET_COMPILE_FLAGS})
target_link_libraries(avk-core 
  PUBLIC ${avk_core_libraries_public}
  PRIVATE ${avk_core_libraries_private}
)
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
