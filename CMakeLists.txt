cmake_minimum_required(VERSION 4.1.0)

# remove old policies (cmake 4 should already do that for us)
# ...

# some common properties
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# project declaration
# default enabled languages: C and CXX, if toolchain doesn't define them https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html#variables-and-properties, automatic
project(AetherVk
  VERSION 0.0.1) 

include(cmake/AvkDetectPlatform.cmake)

# detect used architecture and setup variable
avk_detect_platform()
message(STATUS "Arch=${AVK_ARCH} OS=${AVK_OS} Compiler=${AVK_COMPILER}")

# set CXX flags and define per target (warning) flags
avk_cxx_flags()

# code
add_subdirectory(src)
