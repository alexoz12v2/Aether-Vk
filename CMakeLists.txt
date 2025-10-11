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

# define AVK_DEBUG if necessary
string(TOUPPER ${CMAKE_BUILD_TYPE} AVK_BUILD_FRAGMENT)
set("AVK_${AVK_BUILD_FRAGMENT}" ON)

if((AVK_OS STREQUAL "MACOS") OR (AVK_OS STREQUAL "IOS")) # IOS still TODO
  enable_language(OBJC)
  enable_language(OBJCXX)
endif()

# TODO check for android
avk_setup_vulkan()

# dependencies
avk_setup_vcpkg()
include(${VCPKG_TOOLCHAIN_FILE})
avk_setup_dependencies()

# set CXX flags and define per target (warning) flags
avk_cxx_flags()

# code
add_subdirectory(src)
