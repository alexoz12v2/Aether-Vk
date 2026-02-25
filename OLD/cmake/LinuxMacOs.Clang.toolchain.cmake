# A generic cross-compilation toolchain file
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=generic-toolchain.cmake \
#         -DTARGET_ABI=aarch64-linux-gnu \
#         -DSYSROOT=/path/to/sysroot \
#         ..

# ---- Inputs ----
# TARGET_ABI: e.g. x86_64-linux-gnu, aarch64-linux-gnu, armv7-linux-gnueabihf, x86_64-apple-darwin, arm64-apple-darwin
include_guard()
if(NOT DEFINED TARGET_ABI)
    message(FATAL_ERROR "You must provide -DTARGET_ABI=<triple>")
endif()

# Optional sysroot
if(DEFINED SYSROOT)
    set(CMAKE_SYSROOT ${SYSROOT} CACHE PATH "Target sysroot")
endif()

# ---- System name ----
# CMake uses SYSTEM_NAME to decide platform rules
if(TARGET_ABI MATCHES ".*-linux-.*")
    set(CMAKE_SYSTEM_NAME Linux)
elseif(TARGET_ABI MATCHES ".*-apple-darwin.*")
    set(CMAKE_SYSTEM_NAME Darwin)
else()
    message(FATAL_ERROR "Unknown OS in TARGET_ABI: ${TARGET_ABI}")
endif()

# ---- Processor ----
if(TARGET_ABI MATCHES "^x86_64")
    set(CMAKE_SYSTEM_PROCESSOR x86_64)
elseif(TARGET_ABI MATCHES "^(armv7|arm-)")
    set(CMAKE_SYSTEM_PROCESSOR armv7)
elseif(TARGET_ABI MATCHES "^(aarch64|arm64)")
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
elseif(TARGET_ABI MATCHES "^i[3-6]86")
    set(CMAKE_SYSTEM_PROCESSOR x86)
else()
    message(FATAL_ERROR "Unknown CPU architecture in TARGET_ABI: ${TARGET_ABI}")
endif()

# ---- Compilers ----
find_program(CLANG clang REQUIRED)
find_program(CLANGXX clang++ REQUIRED)

if(CLANG AND CLANGXX)
    # Use clang with --target=<triple>
    set(CMAKE_C_COMPILER ${CLANG})
    set(CMAKE_CXX_COMPILER ${CLANGXX})
    set(CMAKE_C_COMPILER_TARGET ${TARGET_ABI})
    set(CMAKE_CXX_COMPILER_TARGET ${TARGET_ABI})
endif()

# ---- Sysroot handling ----
if(DEFINED CMAKE_SYSROOT)
    set(CMAKE_SYSROOT ${CMAKE_SYSROOT})
endif()

# ---- Path lookup policy ----
# When cross compiling:
# - Programs: look on the host
# - Libraries/Includes: look in the target sysroot
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
