cmake_minimum_required(VERSION 3.25)
include_guard()

function(avk_detect_platform)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)")
        set(AVK_ARCH X86_64)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i[3-6]86|x86)")
        set(AVK_ARCH X86)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv7|armv6|arm[^6])")
        set(AVK_ARCH ARM)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
        set(AVK_ARCH ARM64)
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

    if((CMAKE_C_COMPILER_ID STREQUAL "Clang") OR (CMAKE_C_COMPILER_ID STREQUAL "AppleClang"))
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
# TODO: When using sanitizers, add -fno-optimize-sibling-calls and -fno-omit-frame-pointers
# TODO: When using sanitizers, use dsymutils on MachO Executable to stuff debug information inside it
macro(avk_cxx_flags)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
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
            #string(APPEND CMAKE_CXX_FLAGS " -march=x86-64-v3")
            string(APPEND CMAKE_CXX_FLAGS "")
        elseif(AVK_ARCH STREQUAL "ARMv7A")
            # you need to also check at runtime if you support neon
            string(APPEND CMAKE_CXX_FLAGS " -mfpu=neon")
        elseif(AVK_ARCH STREQUAL "ARMv8A")
            # Usually nothing needed for AArch64, NEON is baseline
            # If targeting AArch32 (32-bit), you might add: -mfpu=neon
        endif()

        # debug only
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            if(${AVK_USE_SANITIZERS})
                # On Windows, you cannot safely mix AddressSanitizer with the MSVC debug CRT.
                # The debug CRT has its own heap instrumentation, so ASan double-hooks the allocator and immediately crashes.
                # set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL") # already done at beg
                if(AVK_OS STREQUAL "LINUX")
                    string(APPEND CMAKE_CXX_FLAGS " -fsanitize=dataflow -fsanitize=thread -fsanitize=memory -fsanitize=leak")
                    if((AVK_ARCH STREQUAL "X86") OR (AVK_ARCH STREQUAL "X86_64"))
                        string(APPEND CMAKE_CXX_FLAGS " -fsanitize=safe-stack")
                    endif()
                elseif(AVK_OS STREQUAL "MACOS")
                    string(APPEND CMAKE_CXX_FLAGS " -fsanitize=thread -fsanitize=leak")
                endif()
                string(APPEND CMAKE_CXX_FLAGS " -fsanitize=address -fsanitize=undefined")
                string(APPEND CMAKE_CXX_FLAGS " -flto -fsanitize=cfi") 

                if(AVK_OS STREQUAL "WINDOWS")
                    # Windows needs to declare the sanitizers DLL if we link against dynamically
                    # Take CMAKE_CXX_COMPILER, which is clang, assume it is a fullpath, go up one level and add \lib\clang 
                    cmake_path(SET AVK_WIN_ASAN_PATH "${CMAKE_CXX_COMPILER}")
                    cmake_path(REMOVE_FILENAME AVK_WIN_ASAN_PATH OUTPUT_VARIABLE AVK_WIN_ASAN_PATH) # -> LLVM/bin/
                    string(REGEX REPLACE "/$" "" AVK_WIN_ASAN_PATH "${AVK_WIN_ASAN_PATH}") # LLVM/bin
                    cmake_path(GET AVK_WIN_ASAN_PATH PARENT_PATH AVK_WIN_ASAN_PATH)  # -> LLVM
                    string(REGEX REPLACE "/$" "" AVK_WIN_ASAN_PATH "${AVK_WIN_ASAN_PATH}") # just to be sure
                    cmake_path(APPEND AVK_WIN_ASAN_PATH "lib/clang")                 # -> LLVM/lib/clang
                    # then list subdirectories and take the highest version
                    # then go to \lib\windows and import shared target having
                    execute_process(
                        COMMAND powershell -NoProfile -ExecutionPolicy Bypass -Command
                            "(Get-ChildItem '${AVK_WIN_ASAN_PATH}' | Sort-Object -Descending)[0].FullName + '\\lib\\windows'"
                        OUTPUT_VARIABLE AVK_WIN_ASAN_PATH
                        ERROR_VARIABLE ERR
                        RESULT_VARIABLE RES
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                    )
                    if ((NOT RES EQUAL 0) OR (NOT IS_DIRECTORY ${AVK_WIN_ASAN_PATH}))
                        message(FATAL_ERROR "${RES}  Couldn't get the path to ASan/UbSan DLLs: ${ERR} (${AVK_WIN_ASAN_PATH})")
                    endif()
                    unset(RES)
                    unset(ERR)
                    message(STATUS "Found Path to ASan/UbSan on windows at ${AVK_WIN_ASAN_PATH}")
                    # DLL: clang_rt.asan_dynamic-x86_64.dll
                    # interface lib: clang_rt.asan_dynamic-x86_64.lib
                    # static lib dependencies: clang_rt.asan_dynamic_runtime_thunk-x86_64.lib
                    # WARN: Linking against this will NOT copy the DLL on the current binary directory!
                    add_library(win-asan-dynamic SHARED IMPORTED)
                    set_target_properties(win-asan-dynamic PROPERTIES
                        IMPORTED_LOCATION "${AVK_WIN_ASAN_PATH}/clang_rt.asan_dynamic-x86_64.dll"
                        IMPORTED_IMPLIB "${AVK_WIN_ASAN_PATH}/clang_rt.asan_dynamic-x86_64.lib"
                        INTERFACE_LINK_LIBRARIES "${AVK_WIN_ASAN_PATH}/clang_rt.asan_dynamic_runtime_thunk-x86_64.lib"
                    )
                endif()
            else()
                set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDebugDLL")
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


function(avk_setup_dependencies)
    if(NOT TARGET Boost::context)
        find_package(Boost REQUIRED COMPONENTS context)
    endif()
    message(WARNING "define environment variable before build: VCPKG_ROOT = \"${VCPKG_ROOT}\"")
endfunction()


function(avk_setup_vcpkg)
    if((IS_DIRECTORY ${CMAKE_BINARY_DIR}/vcpkg) AND DEFINED VCPKG_COMMAND AND DEFINED VCPKG_ROOT)
        execute_process(
            COMMAND ${VCPKG_COMMAND} help
            RESULT_VARIABLE RES
        )
        if(NOT RES EQUAL 0)
            message(FATAL_ERROR "repository vcpkg installation invalid. delete the build/vkpkg and build/vcpkg_installed dirs")
        endif()
    else()
        message(STATUS "Fetching vcpkg...")
        execute_process(
            COMMAND git clone https://github.com/microsoft/vcpkg.git
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            RESULT_VARIABLE RES
            ECHO_OUTPUT_VARIABLE
        )
        if(NOT RES EQUAL 0)
            message(FATAL_ERROR "Could not clone vcpkg")
        endif()

        if(WIN32)
            execute_process(
                COMMAND cmd /c ${CMAKE_BINARY_DIR}/vcpkg/bootstrap-vcpkg.bat
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/vcpkg
                RESULT_VARIABLE RES
                ECHO_OUTPUT_VARIABLE
            )
        else()
            execute_process(
                COMMAND ${CMAKE_BINARY_DIR}/vcpkg/bootstrap-vcpkg.sh
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/vcpkg
                RESULT_VARIABLE RES
                ECHO_OUTPUT_VARIABLE
            )
        endif()
        if(NOT RES EQUAL 0)
            message(FATAL_ERROR "Couldn't download vcpkg executable")
        endif()

        set(VCPKG_ROOT "${CMAKE_BINARY_DIR}/vcpkg" CACHE STRING "env var for vcpkg" FORCE)
        set($ENV{VCPKG_ROOT} ${VCPKG_ROOT}) # probably won't work
        if(WIN32)
            set(VCPKG_COMMAND ${CMAKE_BINARY_DIR}/vcpkg/vcpkg.exe CACHE STRING "vcpkg executable" FORCE)
        else()
            set(VCPKG_COMMAND ${CMAKE_BINARY_DIR}/vcpkg/vcpkg CACHE STRING "vcpkg executable" FORCE)
        endif()

        # to include outside of a function context such that dependencies from vcpkg can be handled
        # VCPKG_CHAINLOAD_TOOLCHAIN_FILE is used as a cache variable to propagate information to main build
        # so use another name
        set(VCPKG_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "toolchain file" FORCE)
    endif()
    return(PROPAGATE VCPKG_TOOLCHAIN_FILE VCPKG_COMMAND VCPKG_ROOT)
endfunction()

