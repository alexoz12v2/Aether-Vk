############################################################################
# Automatically collect headers and sources
function(collect_os_specific_files VAR_NAME SRC_DIR OS PUBLIC_OR_PRIVATE)
  # Recurse all files in the given folder
  if (PUBLIC_OR_PRIVATE STREQUAL "public")
    file(GLOB_RECURSE ALL_FILES CONFIGURE_DEPENDS
      "${SRC_DIR}/${PUBLIC_OR_PRIVATE}/*.h"
      "${SRC_DIR}/${PUBLIC_OR_PRIVATE}/*.hpp"
    )
  elseif (PUBLIC_OR_PRIVATE STREQUAL "private")
    set(FILESPECS
      "${SRC_DIR}/${PUBLIC_OR_PRIVATE}/*.cpp"
      "${SRC_DIR}/${PUBLIC_OR_PRIVATE}/*.cc"
      "${SRC_DIR}/${PUBLIC_OR_PRIVATE}/*.cxx"
    )
    if (AVK_OS STREQUAL "MACOS" OR AVK_OS STREQUAL "IOS")
      list(APPEND FILESPECS
        "${SRC_DIR}/${PUBLIC_OR_PRIVATE}/*.m"
        "${SRC_DIR}/${PUBLIC_OR_PRIVATE}/*.mm"
      )
    endif ()
    file(GLOB_RECURSE ALL_FILES CONFIGURE_DEPENDS ${FILESPECS})
  else ()
    message(FATAL_ERROR "Unexpected folder name")
  endif ()

  # Separate OS-specific files
  set(OS_FILES ${ALL_FILES})
  list(FILTER ALL_FILES EXCLUDE REGEX "/${PUBLIC_OR_PRIVATE}/os/")

  if (${OS} STREQUAL "WINDOWS")
    list(FILTER OS_FILES INCLUDE REGEX "/${PUBLIC_OR_PRIVATE}/os(/windows)?/.*\\.(h|hpp|cpp)$")
  elseif (${OS} STREQUAL "LINUX")
    list(FILTER OS_FILES INCLUDE REGEX "/${PUBLIC_OR_PRIVATE}/os(/linux|/posix)?/.*\\.(h|hpp|cpp)$")
  elseif (${OS} STREQUAL "MACOS")
    list(FILTER OS_FILES INCLUDE REGEX "/${PUBLIC_OR_PRIVATE}/os(/macos|/apple|/posix)?/.*\\.(h|hpp|cpp|m|mm)$")
  elseif (${OS} STREQUAL "ANDROID")
    list(FILTER OS_FILES INCLUDE REGEX "/${PUBLIC_OR_PRIVATE}/os(/android|/posix)/.*\\.(h|hpp|cpp)$")
  elseif (${OS} STREQUAL "IOS")
    list(FILTER OS_FILES INCLUDE REGEX "/${PUBLIC_OR_PRIVATE}/os(/ios|/apple|/posix)/.*\\.(h|hpp|cpp)$")
  else ()
    # default: only direct children
    list(FILTER OS_FILES INCLUDE REGEX "/${PUBLIC_OR_PRIVATE}/os/.*\\.(h|hpp|cpp)$")
  endif ()

  # Merge all files
  set(FINAL_LIST ${ALL_FILES} ${OS_FILES})

  # Propagate result
  set(${VAR_NAME} ${FINAL_LIST} PARENT_SCOPE)
endfunction()

collect_os_specific_files(AVK_CORE_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR} ${AVK_OS} public)
collect_os_specific_files(AVK_CORE_PRIVATE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR} ${AVK_OS} private)

# TODO Remove debug logging
foreach (item ${AVK_CORE_PRIVATE_SOURCES})
  message(STATUS " src: ${item}")
endforeach ()
foreach (item ${AVK_CORE_PUBLIC_HEADERS})
  message(STATUS " hrd: ${item}")
endforeach ()
############################################################################

set(avk_core_libraries_public
  Boost::fiber Vulkan::Headers GPUOpen::VulkanMemoryAllocator volk::volk_headers
  glm::glm-header-only
)
set(avk_core_libraries_private KTX::ktx imageinfo::imageinfo)
if ((AVK_OS STREQUAL "MACOS") OR (AVK_OS STREQUAL "IOS"))
  # Volk loads MoltenVK at runtime! Just bundle it.
  # list(APPEND avk_core_libraries_public "${MVK_DYLIB}")
  list(APPEND avk_core_libraries_private
    "-framework Cocoa"
    "-framework Foundation"
    "-framework Metal"
    "-framework QuartzCore"
    "-framework IOSurface"
    "-framework IOKit"
    "-framework CoreGraphics"
    "-framework CoreServices"
  )
endif ()
if (AVK_OS STREQUAL "WINDOWS")
  list(APPEND avk_core_libraries_private dwmapi Dbghelp)
endif ()
if (AVK_OS STREQUAL "ANDROID")
  list(APPEND avk_core_libraries_public android log)
endif ()

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
  CMAKE_CURRENT_SOURCE_DIR}/public
  PRIVATE
  ${Stb_INCLUDE_DIR}
  ${AVK_TINYGLTF_INCLUDE_DIR}
)

add_library(avk::core ALIAS avk-core)
