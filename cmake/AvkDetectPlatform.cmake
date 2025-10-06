cmake_minimum_required(VERSION 3.25)
include_guard()

function(avk_detect_platform)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)")
        set(AVK_ARCH X86_64)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i[3-6]86|x86)")
        set(AVK_ARCH X86)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv7|arm)")
        set(AVK_ARCH ARMv7A)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
        set(AVK_ARCH ARMv8A)
    else()
        set(AVK_ARCH UNKNOWN)
    endif()

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(AVK_OS LINUX)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(AVK_OS MACOS)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(AVK_OS WINDOWS)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
        set(AVK_OS ANDROID)
    else()
        set(AVK_OS UNKNOWN)
    endif()

    if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
        set(AVK_COMPILER CLANG)
    elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        set(AVK_COMPILER GCC)
    elseif(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
        set(AVK_COMPILER MSVC)
    else()
        set(AVK_COMPILER UNKNOWN)
    endif()

    return(PROPAGATE AVK_ARCH AVK_OS AVK_COMPILER)
endfunction()


# assumes avk_detect_platform was called
# return
# - AVK_CXX_TARGET_COMPILE_FLAGS space separated list of compile flags (warnings and such)
macro(avk_cxx_flags)
    if(AVK_COMPILER STREQUAL "CLANG")
        # general
        string(APPEND CMAKE_CXX_FLAGS " -fvisibility=hidden -fno-fast-math -faddrsig -fstrict-aliasing")
        string(APPEND CMAKE_CXX_FLAGS " -nogpuinc -nogpulib -fstack-protector-all")

        # os specific
        if(AVK_OS STREQUAL "LINUX" OR AVK_OS STREQUAL "MACOS")
            string(APPEND CMAKE_CXX_FLAGS " -pthread -stdlib=libc++")
        endif()

        # arch specific
        if(AVK_ARCH STREQUAL "X86_64")
            string(APPEND CMAKE_CXX_FLAGS " -march=x86-64-v3")
        elseif(AVK_ARCH STREQUAL "ARMv7A")
            # you need to also check at runtime if you support neon
            string(APPEND CMAKE_CXX_FLAGS " -mfpu=neon")
        elseif(AVK_ARCH STREQUAL "ARMv8A")
            # Usually nothing needed for AArch64, NEON is baseline
            # If targeting AArch32 (32-bit), you might add: -mfpu=neon
        endif()

        # debug only
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            if(NOT AVK_OS STREQUAL "WINDOWS" AND NOT AVK_OS STREQUAL "ANDROID")
                string(APPEND CMAKE_CXX_FLAGS " -fsanitize=dataflow -fsanitize=thread -fsanitize=memory -fsanitize=safe-stack") 
            endif()

            if(${AVK_USE_SANITIZERS})
                string(APPEND CMAKE_CXX_FLAGS " -fsanitize=address -fsanitize=undefined")
                string(APPEND CMAKE_CXX_FLAGS " -flto -fsanitize=cfi") 
            else()
                string(APPEND CMAKE_CXX_FLAGS " -flto") 
            endif()

            string(APPEND CMAKE_CXX_FLAGS " -g -O0")
        elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
            string(APPEND CMAKE_CXX_FLAGS " -flto -fwhole-program-vtables -O2")
        elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
            string(APPEND CMAKE_CXX_FLAGS " -g -flto -fwhole-program-vtables -O2")
        elseif(CMAKE_BUILG_TYPE STREQUAL "MinSizeRel")
            string(APPEND CMAKE_CXX_FLAGS " -Os -freroll-loops")
        else() # Release
            string(APPEND CMAKE_CXX_FLAGS " -flto -fwhole-program-vtables -O2")
        endif()

        # return target specific flags
        set(AVK_CXX_TARGET_COMPILE_FLAGS "-Wall;-Wextra;-pedantic;-Werror")
    else()
        message(FATAL_ERROR "Only Clang Toolchain supported")
    endif()
endmacro()


macro(avk_setup_vulkan)
    if(NOT DEFINED ENV{VULKAN_SDK})
        message(FATAL_ERROR "VULKAN_SDK Environment Variable not defined. Please install Vulkan SDK >= 1.4 and define that")
    endif()

    # Note: Should we rely on installed SDK or download it like we do with bazel?
    find_package(Vulkan REQUIRED) # Imported Target: Vulkan::Vulkan
    set(SLANGC_COMMAND "${Vulkan_GLSLC_EXECUTABLE}")

    if(WIN32)
        cmake_path(REPLACE_FILENAME SLANGC_COMMAND "slangc.exe")
    else()
        cmake_path(REPLACE_FILENAME SLANGC_COMMAND "slangc")
    endif()

    execute_process(COMMAND ${SLANGC_COMMAND} "--help" RESULT_VARIABLE SLANG_RES OUTPUT_QUIET ERROR_QUIET)
    if (NOT SLANG_RES EQUAL 0)
        message(FATAL_ERROR "Couldn't find slangc executable at \"${SLANGC_COMMAND}\", check Vulkan"
            " SDK Version and possibly upgrade it.")
    endif()
    unset(SLANG_RES)

    cmake_path(REMOVE_FILENAME SLANGC_COMMAND OUTPUT_VARIABLE VULKAN_BIN_PATH)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink "${VULKAN_BIN_PATH}" vulkan-sdk
        RESULT_VARIABLE VULKAN_SYMLINK_RES)
    if(NOT VULKAN_SYMLINK_RES EQUAL 0)
        message(FATAL_ERROR "Failed to create Symlink to Vulkan SDK bin directory inside the current directory")
    endif()
    unset(VULKAN_SYMLINK_RES)
    unset(VULKAN_BIN_PATH)
endmacro()
