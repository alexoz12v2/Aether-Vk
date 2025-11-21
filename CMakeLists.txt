cmake_minimum_required(VERSION 4.1.0)

# remove old policies (cmake 4 should already do that for us)
# ...

# some common properties
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# cache variables
option(AVK_USE_SANITIZERS "Use sanitizers during non-release compilations with LLVM toolchain")

# project declaration
# default enabled languages: C and CXX, if toolchain doesn't define them https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html#variables-and-properties, automatic
project(AetherVk
  VERSION 0.0.1)

enable_language(ASM)

include(cmake/AvkDetectPlatform.cmake)

# detect used architecture and setup variable
avk_detect_platform()
message(STATUS "Arch=${AVK_ARCH} OS=${AVK_OS} Compiler=${AVK_COMPILER}")
# iOS/macOS abis manually declared here
if ((AVK_OS STREQUAL "MACOS" OR AVK_OS STREQUAL "IOS") AND NOT DEFINED CMAKE_OSX_ARCHITECTURES)
  # set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "Target ABIs for iOS/macOS")
  message(FATAL_ERROR "You need to set properly the 'CMAKE_OSX_ARCHITECTURES' variable")
endif ()


# convenience symlink for android
if (AVK_OS STREQUAL "ANDROID" AND DEFINED ANDROID_NDK)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${ANDROID_NDK} ${CMAKE_SOURCE_DIR}/build-ndk-symlink
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${ANDROID_NDK} ${CMAKE_SOURCE_DIR}/src/launcher/android/build-ndk-symlink
  )
endif ()

# define AVK_DEBUG if necessary
string(TOUPPER ${CMAKE_BUILD_TYPE} AVK_BUILD_FRAGMENT)
set("AVK_${AVK_BUILD_FRAGMENT}" ON)

if ((AVK_OS STREQUAL "MACOS") OR (AVK_OS STREQUAL "IOS"))
  enable_language(OBJC)
  enable_language(OBJCXX)
endif ()

# TODO check for android
avk_setup_vulkan()
avk_create_vulkan_sdk_library_targets()

# dependencies
avk_setup_vcpkg()
if (AVK_OS STREQUAL "MACOS")
  list(LENGTH CMAKE_OSX_ARCHITECTURES osx_list_length)
  if (osx_list_length GREATER 1)
    message(STATUS "Assuming you are building macOS Universal Binaries")
    execute_process(
      COMMAND sh -c "${CMAKE_CXX_COMPILER} --version | awk 'FNR <= 1 {print $1}'"
      OUTPUT_VARIABLE AVK_CompilerVendor
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY
    )
    if (NOT AVK_CompilerVendor STREQUAL "Apple")
      message(FATAL_ERROR "For a Universal Binary compilation, Apple Clang is the only thing that works. Found: ${AVK_CompilerVendor}")
    endif ()
    set(VCPKG_MANIFEST_INSTALL OFF CACHE BOOL "vkpkg call install on toolchain" FORCE)
  else ()
    set(VCPKG_MANIFEST_INSTALL ON CACHE BOOL "vkpkg call install on toolchain" FORCE)
  endif ()
endif ()
include("${VCPKG_TOOLCHAIN_FILE}")
avk_setup_dependencies()

# set CXX flags and define per target (warning) flags
avk_cxx_flags()

# static shaders target
file(GLOB_RECURSE avk_shader_file_list "${CMAKE_SOURCE_DIR}/shaders/*.spv")
message(STATUS "${avk_shader_file_list}")
# TODO better
# add_custom_target(avk-copy-static-shaders ALL 
#   COMMAND ${CMAKE_COMMAND} -E copy ${avk_shader_file_list} ${CMAKE_BINARY_DIR}
# )
# TODO better
add_custom_target(avk-copy-static-shaders ALL
  COMMAND ${CMAKE_COMMAND} -E copy ${avk_shader_file_list} ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(avk-copy-assets ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/assets
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets ${CMAKE_CURRENT_BINARY_DIR}/assets
)

# code
add_subdirectory(src)
