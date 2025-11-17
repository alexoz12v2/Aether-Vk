cmake_minimum_required(VERSION 3.25)
include_guard()

function(avk_detect_platform)
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)")
    set(AVK_ARCH X86_64)
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(i[3-6]86|x86)")
    set(AVK_ARCH X86)
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv7|armv6|arm[^6])")
    set(AVK_ARCH ARM)
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
    set(AVK_ARCH ARM64)
  else ()
    set(AVK_ARCH UNKNOWN)
  endif ()

  # https://cmake.org/cmake/help/latest/variable/CMAKE_SYSTEM_NAME.html#variable:CMAKE_SYSTEM_NAME
  if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(AVK_OS "LINUX")
    set(AVK_POSIX ON)
  elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(AVK_OS "MACOS")
    set(AVK_POSIX ON)
  elseif (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(AVK_OS "IOS")
    set(AVK_POSIX ON)
  elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(AVK_OS "WINDOWS")
    set(AVK_POSIX OFF)
  elseif (CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(AVK_OS "ANDROID")
    set(AVK_POSIX ON)
  else ()
    set(AVK_OS UNKNOWN)
    set(AVK_POSIX OFF)
  endif ()

  # TODO: detect features with cmake -p (wrapped in cmake shipped modules). Example
  #######################################################################
  # --- Step 2: Use built-in CMake modules to check for features ---
  # include(CheckIncludeFile)
  # include(CheckIncludeFiles)
  # include(CheckFunctionExists)
  # include(CheckSymbolExists)
  # include(CheckTypeSize)
  #
  # # These will set HAVE_xxx variables automatically
  # check_include_file("unistd.h" HAVE_UNISTD_H)
  # check_include_file("pthread.h" HAVE_PTHREAD_H)
  # check_include_file("sys/mman.h" HAVE_SYS_MMAN_H)
  # check_include_file("signal.h" HAVE_SIGNAL_H)
  #
  # check_function_exists(mmap HAVE_MMAP)
  # check_function_exists(sigaction HAVE_SIGACTION)
  # check_function_exists(clock_gettime HAVE_CLOCK_GETTIME)
  # check_function_exists(gettimeofday HAVE_GETTIMEOFDAY)
  # check_function_exists(pthread_create HAVE_PTHREAD_CREATE)
  # check_function_exists(fork HAVE_FORK)
  #
  # # --- Step 3: Generate a configuration header from a template ---
  # # Create a template file: platform_config.h.in
  # configure_file(
  #   ${CMAKE_CURRENT_SOURCE_DIR}/platform_config.h.in
  #   ${CMAKE_CURRENT_BINARY_DIR}/platform_config.h
  # )
  ## where the configuration file has this form: ---------------
  # #pragma once
  #
  #/* OS identifiers */
  ##define AVK_OS "@AVK_OS@"
  #
  #/* POSIX flag */
  ##cmakedefine AVK_POSIX 1
  #
  #/* Feature availability (checked by CMake) */
  ##cmakedefine HAVE_UNISTD_H 1
  ##cmakedefine HAVE_PTHREAD_H 1
  ##cmakedefine HAVE_SYS_MMAN_H 1
  ##cmakedefine HAVE_SIGNAL_H 1
  #
  ##cmakedefine HAVE_MMAP 1
  ##cmakedefine HAVE_SIGACTION 1
  ##cmakedefine HAVE_CLOCK_GETTIME 1
  ##cmakedefine HAVE_GETTIMEOFDAY 1
  ##cmakedefine HAVE_PTHREAD_CREATE 1
  ##cmakedefine HAVE_FORK 1
  ## ----------------------------
  # # --- Step 4: Add your target and include the generated config ---
  # add_executable(example main.cpp)
  # target_include_directories(example PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  #######################################################################

  if ((CMAKE_C_COMPILER_ID STREQUAL "Clang") OR (CMAKE_C_COMPILER_ID STREQUAL "AppleClang"))
    set(AVK_COMPILER CLANG)
  elseif (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(AVK_COMPILER GCC)
  elseif (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    set(AVK_COMPILER MSVC)
  else ()
    set(AVK_COMPILER UNKNOWN)
  endif ()

  return(PROPAGATE AVK_ARCH AVK_OS AVK_COMPILER AVK_POSIX)
endfunction()


# assumes avk_detect_platform was called
# return
# - AVK_CXX_TARGET_COMPILE_FLAGS space separated list of compile flags (warnings and such)
# TODO: When using sanitizers, add -fno-optimize-sibling-calls and -fno-omit-frame-pointers
# TODO: When using sanitizers, use dsymutils on MachO Executable to stuff debug information inside it
# TODO: When using sanitizers (cfi), volk breaks
# TODO: Add no-lto and no-sanitizer-cfi cmake cache variables
# TODO: Make this per target?
macro(avk_cxx_flags)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
  if (AVK_COMPILER STREQUAL "CLANG")
    # general
    string(APPEND CMAKE_CXX_FLAGS " -fvisibility=hidden -fno-fast-math -faddrsig -fstrict-aliasing")
    string(APPEND CMAKE_CXX_FLAGS " -nogpuinc -nogpulib -fstack-protector-all")

    # os specific
    if (AVK_OS STREQUAL "LINUX" OR AVK_OS STREQUAL "MACOS")
      string(APPEND CMAKE_CXX_FLAGS " -pthread -stdlib=libc++")
    endif ()

    # arch specific
    if (AVK_ARCH STREQUAL "X86_64")
      #string(APPEND CMAKE_CXX_FLAGS " -march=x86-64-v3")
      string(APPEND CMAKE_CXX_FLAGS "")
    elseif (AVK_ARCH STREQUAL "ARMv7A")
      # you need to also check at runtime if you support neon
      string(APPEND CMAKE_CXX_FLAGS " -mfpu=neon")
    elseif (AVK_ARCH STREQUAL "ARMv8A")
      # Usually nothing needed for AArch64, NEON is baseline
      # If targeting AArch32 (32-bit), you might add: -mfpu=neon
    endif ()

    # debug only
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
      if (${AVK_USE_SANITIZERS})
        # On Windows, you cannot safely mix AddressSanitizer with the MSVC debug CRT.
        # The debug CRT has its own heap instrumentation, so ASan double-hooks the allocator and immediately crashes.
        # set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL") # already done at beg
        if (AVK_OS STREQUAL "LINUX")
          string(APPEND CMAKE_CXX_FLAGS " -fsanitize=dataflow -fsanitize=thread -fsanitize=memory -fsanitize=leak")
          if ((AVK_ARCH STREQUAL "X86") OR (AVK_ARCH STREQUAL "X86_64"))
            string(APPEND CMAKE_CXX_FLAGS " -fsanitize=safe-stack")
          endif ()
        elseif (AVK_OS STREQUAL "MACOS")
          string(APPEND CMAKE_CXX_FLAGS " -fsanitize=thread -fsanitize=leak")
        endif ()

        # Common sanitizers which should work (on desktop at least)
        if (NOT (AVK_OS STREQUAL "ANDROID") AND NOT (AVK_OS STREQUAL "IOS"))
          string(APPEND CMAKE_CXX_FLAGS " -fsanitize-ignorelist=${CMAKE_SOURCE_DIR}/blacklist.sanitizers")
          string(APPEND CMAKE_CXX_FLAGS " -fsanitize=address -fsanitize=undefined")
          # Control Flow Integrity Requires LTO -> (Windows, clang 21) LLVM ERROR: Associative COMDAT symbol '___asan_gen_anon_global' is not a key for its COMDAT.
          # -fsanitize=cfi-vcall -> CRASH
          # -fsanitize=cfi-mfcall -> doesn't exist on windows
          # string(APPEND CMAKE_CXX_FLAGS " -flto -fsanitize=cfi")
          string(APPEND CMAKE_CXX_FLAGS " -flto -fsanitize=cfi-cast-strict -fsanitize=cfi-nvcall -fsanitize=cfi-icall -fsanitize=cfi-derived-cast -fsanitize=cfi-unrelated-cast")
        endif ()

        # windows has to copy the DLLs as usual
        if (AVK_OS STREQUAL "WINDOWS")
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
          endif ()
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
        endif ()
      else ()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDebugDLL")
        string(APPEND CMAKE_CXX_FLAGS " -flto")
      endif ()

      string(APPEND CMAKE_CXX_FLAGS " -g -O0")
    elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
      string(APPEND CMAKE_CXX_FLAGS " -flto -fwhole-program-vtables -O2")
    elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
      string(APPEND CMAKE_CXX_FLAGS " -g -flto -fwhole-program-vtables -O2")
    elseif (CMAKE_BUILG_TYPE STREQUAL "MinSizeRel")
      string(APPEND CMAKE_CXX_FLAGS " -Os -freroll-loops")
    else () # Release
      string(APPEND CMAKE_CXX_FLAGS " -flto -fwhole-program-vtables -O2")
    endif ()

    # return target specific flags
    set(AVK_CXX_TARGET_COMPILE_FLAGS "-Wall;-Wextra;-pedantic;-Werror")
  else ()
    message(FATAL_ERROR "Only Clang Toolchain supported")
  endif ()
endmacro()


macro(avk_setup_vulkan)
  if (NOT DEFINED ENV{VULKAN_SDK})
    message(FATAL_ERROR "VULKAN_SDK Environment Variable not defined. Please install Vulkan SDK >= 1.4 and define that")
  endif ()

  # Note: Should we rely on installed SDK or download it like we do with bazel?
  if (AVK_OS STREQUAL "MACOS")
    find_package(Vulkan REQUIRED COMPONENTS MoltenVK)
  else ()
    find_package(Vulkan REQUIRED)
  endif ()
  set(SLANGC_COMMAND "${Vulkan_GLSLC_EXECUTABLE}")

  if (WIN32)
    cmake_path(REPLACE_FILENAME SLANGC_COMMAND "slangc.exe")
  else ()
    cmake_path(REPLACE_FILENAME SLANGC_COMMAND "slangc")
  endif ()

  execute_process(COMMAND ${SLANGC_COMMAND} "--help" RESULT_VARIABLE SLANG_RES OUTPUT_QUIET ERROR_QUIET)
  if (NOT SLANG_RES EQUAL 0)
    message(FATAL_ERROR "Couldn't find slangc executable at \"${SLANGC_COMMAND}\", check Vulkan"
      " SDK Version and possibly upgrade it.")
  endif ()
  unset(SLANG_RES)

  cmake_path(REMOVE_FILENAME SLANGC_COMMAND OUTPUT_VARIABLE VULKAN_BIN_PATH)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${VULKAN_BIN_PATH}" "${CMAKE_SOURCE_DIR}/vulkan-sdk"
    RESULT_VARIABLE VULKAN_SYMLINK_RES)
  if (NOT VULKAN_SYMLINK_RES EQUAL 0)
    message(FATAL_ERROR "Failed to create Symlink to Vulkan SDK bin directory inside the current directory")
  endif ()
  unset(VULKAN_SYMLINK_RES)
  unset(VULKAN_BIN_PATH)
  # prevent header overrides by vcpkg creating a target alias
  add_library(VulkanSDK::Headers ALIAS Vulkan::Headers)
endmacro()


function(avk_setup_dependencies)
  if (NOT DEFINED VCPKG_COMMAND OR NOT DEFINED VCPKG_ROOT)
    message(FATAL_ERROR "Couldn't find vcpkg on the build directory")
  endif ()

  set(AVK_VCPKG_INSTALL_DIR "${VCPKG_ROOT}_installed/${VCPKG_TARGET_TRIPLET}")
  message(STATUS "To use vcpkg, define environment variable before build: VCPKG_ROOT = \"${VCPKG_ROOT}\" (or use target)")
  add_custom_target(avk-vcpkg COMMAND ${VCKPG_COMMAND})

  # Begin Dependencies Setup: Boost
  if (NOT TARGET Boost::fiber)
    find_package(Boost REQUIRED COMPONENTS fiber)
    if (WIN32 AND ${AVK_USE_SANITIZERS})
      message(WARNING "Remapping Debug Boost to Release to avoid ASan vs ucrtbased.dll conflict")
      avk_release_is_debug_for_imported(Boost::fiber)
      # must do it for all fiber dependencies too
      set(deps assert config context core intrusive predef smart_ptr)
      foreach (dep ${deps})
        if (TARGET "Boost::${dep}")
          avk_release_is_debug_for_imported("Boost::${dep}")
        else ()
          message(WARNING "Boost::${dep}")
        endif ()
      endforeach ()
    endif ()
  endif ()

  if (NOT TARGET volk::volk)
    find_package(volk CONFIG REQUIRED)
  endif ()

  if (NOT TARGET GPUOpen::VulkanMemoryAllocator)
    if (NOT DEFINED Vulkan_LIBRARY)
      message(FATAL_ERROR "Must use FindVulkan.cmake before VulkanMemoryAllocator")
    endif ()
    find_package(VulkanMemoryAllocator CONFIG REQUIRED)
  endif ()

  # -- Math Library --
  if (NOT TARGET glm::glm-header-only)
    find_package(glm CONFIG REQUIRED)
  endif ()

  # -- Image File Formats --
  # fallback: try to decode it with STB if supported
  # usage: target_include_directories(main PRIVATE ${Stb_INCLUDE_DIR})
  if (NOT DEFINED Stb_INCLUDE_DIR)
    find_package(Stb REQUIRED)
  endif ()

  # PNG: libpng
  if (NOT TARGET PNG::PNG)
    find_package(PNG REQUIRED)
    if (WIN32 AND ${AVK_USE_SANITIZERS})
      avk_release_is_debug_for_imported(PNG::PNG)
    endif ()
  endif ()

  if (NOT TARGET KTX::ktx)
    find_package(ktx CONFIG REQUIRED)
    if (WIN32 AND ${AVK_USE_SANITIZERS})
      avk_release_is_debug_for_imported(KTX::ktx)
    endif ()
  endif ()

  # -- scene format: GLTF --
  if (NOT DEFINED AVK_TINYGLTF_INCLUDE_DIR)
    set(AVK_TINYGLTF_INCLUDE_DIR "${AVK_VCPKG_INSTALL_DIR}/include")
  endif ()

  # when using sanitizers, we cannot link against the debug MSVC C Runtime, otherwise, whenever we call something
  # from it, we should note it with no_sanitize. No. prefer linking against the release runtime
  if (WIN32 AND ${AVK_USE_SANITIZERS} AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    # KTX 4.3.2 adds as an interface definition:  $<$<CONFIG:Debug>:_DEBUG;DEBUG>
    # Loop over all targets and remove _DEBUG from interface defs
    foreach (t KTX::ktx PNG::PNG Boost::fiber)
      message(STATUS "FIXING DEBUG FLAGS/DEFINITIONS For Target ${t}")
      get_target_property(defs ${t} INTERFACE_COMPILE_DEFINITIONS)
      message(STATUS "  ${defs}")
      if (defs)
        string(REGEX REPLACE "\\$<\\$<CONFIG:Debug>:_DEBUG;DEBUG>" "$<$<CONFIG:Debug>:DEBUG>" defs "${defs}")
        list(FILTER defs EXCLUDE REGEX "^\\$<\\$<CONFIG:Debug>:_DEBUG>")
        list(FILTER defs EXCLUDE REGEX "_DEBUG")
        message(STATUS "  LIST AFTER ${defs}")
        set_target_properties(${t} PROPERTIES
          INTERFACE_COMPILE_DEFINITIONS "${defs}"
        )
        get_target_property(defs ${t} INTERFACE_COMPILE_DEFINITIONS)
        message(STATUS "  AFTER: ${defs}")
      endif ()
      get_target_property(defs ${t} INTERFACE_COMPILE_OPTIONS)
      message(STATUS "  ${defs}")
      if (defs)
        list(REMOVE_ITEM defs "-MTd")
        list(REMOVE_ITEM defs "-MDd")
        set_target_properties(${t} PROPERTIES
          INTERFACE_COMPILE_OPTIONS "${defs}"
        )
      endif ()
    endforeach ()
  endif ()

  # Note: they are actually the same, just to be sure include both
  return(PROPAGATE Stb_INCLUDE_DIR AVK_TINYGLTF_INCLUDE_DIR)
endfunction()


function(avk_setup_vcpkg)
  if (DEFINED VCPKG_COMMAND AND DEFINED VCPKG_ROOT)
    execute_process(
      COMMAND ${VCPKG_COMMAND} help
      RESULT_VARIABLE RES
    )
    if (NOT RES EQUAL 0)
      message(FATAL_ERROR "repository vcpkg installation invalid. delete the build/vkpkg and build/vcpkg_installed dirs")
    endif ()
  else ()
    message(STATUS "Fetching vcpkg...")
    execute_process(
      COMMAND git clone https://github.com/microsoft/vcpkg.git
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      RESULT_VARIABLE RES
      ECHO_OUTPUT_VARIABLE
    )
    if (NOT RES EQUAL 0)
      message(FATAL_ERROR "Could not clone vcpkg")
    endif ()

    if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
      execute_process(
        COMMAND cmd /c ${CMAKE_BINARY_DIR}/vcpkg/bootstrap-vcpkg.bat
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/vcpkg
        RESULT_VARIABLE RES
        ECHO_OUTPUT_VARIABLE
      )
    else ()
      execute_process(
        COMMAND ${CMAKE_BINARY_DIR}/vcpkg/bootstrap-vcpkg.sh
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/vcpkg
        RESULT_VARIABLE RES
        ECHO_OUTPUT_VARIABLE
      )
    endif ()
    if (NOT RES EQUAL 0)
      message(FATAL_ERROR "Couldn't download vcpkg executable")
    endif ()

    set(VCPKG_ROOT "${CMAKE_BINARY_DIR}/vcpkg" CACHE STRING "env var for vcpkg" FORCE)
    set($ENV{VCPKG_ROOT} ${VCPKG_ROOT}) # probably won't work
    if (WIN32)
      set(VCPKG_COMMAND ${CMAKE_BINARY_DIR}/vcpkg/vcpkg.exe CACHE STRING "vcpkg executable" FORCE)
    else ()
      set(VCPKG_COMMAND ${CMAKE_BINARY_DIR}/vcpkg/vcpkg CACHE STRING "vcpkg executable" FORCE)
    endif ()

    # to include outside of a function context such that dependencies from vcpkg can be handled
    # VCPKG_CHAINLOAD_TOOLCHAIN_FILE is used as a cache variable to propagate information to main build
    # so use another name
    set(VCPKG_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "toolchain file" FORCE)
  endif ()

  # <vcpkg root>/buildsystems/vcpkg.cmake assumes, on windows, that we are not cross compiling. wrong.
  if (AVK_OS STREQUAL "ANDROID")
    # manually set VCPKG_TARGET_TRIPLET variable to override whatever vcpkg figures out
    if (NOT DEFINED ANDROID_ABI)
      message(FATAL_ERROR "ANDROID_ABI should be a cache variable defined by the NDK toolchain file")
    endif ()

    # avoid vcpkg reincluding the android ndk toolchain
    set(ANDROID_NDK_TOOLCHAIN_INCLUDED ON)
    set(ANDROID_USE_LEGACY_TOOLCHAIN_FILE OFF)

    if (NOT DEFINED ANDROID_NDK)
      message(FATAL_ERROR "ANDROID_NDK should be a cache variable defined by the NDK toolchain file")
    else ()
      message(WARNING
        "Defining environment variable ANDROID_NDK_HOME for cmake process. This is "
        "is consumed by vcpkg but the calling process won't have it")
      set(ENV{ANDROID_NDK_HOME} ${ANDROID_NDK})
    endif ()

    if (ANDROID_ABI STREQUAL "arm64-v8a")
      set(VCPKG_TARGET_TRIPLET "arm64-android")
    elseif (ANDROID_ABI STREQUAL "armeabi-v7a")
      set(VCPKG_TARGET_TRIPLET "arm-neon-android")
    elseif (ANDROID_ABI STREQUAL "x86")
      set(VCPKG_TARGET_TRIPLET "x86-android")
    elseif (ANDROID_ABI STREQUAL "x86_64")
      set(VCPKG_TARGET_TRIPLET "x64-android")
    else ()
      message(FATAL_ERROR "Unsupported ANDROID_ABI: ${ANDROID_ABI}")
    endif ()
  endif ()

  set(RETURN_LIST VCPKG_TOOLCHAIN_FILE VCPKG_COMMAND VCPKG_ROOT)
  if (DEFINED VCPKG_TARGET_TRIPLET)
    message(STATUS "Overridden VCPKG_TARGET_TRIPLET: ${VCPKG_TARGET_TRIPLET}")
    list(APPEND RETURN_LIST VCPKG_TARGET_TRIPLET)
  endif ()
  if (DEFINED ANDROID_NDK)
    list(APPEND RETURN_LIST ANDROID_NDK_TOOLCHAIN_INCLUDED ANDROID_USE_LEGACY_TOOLCHAIN_FILE)
  endif ()
  return(PROPAGATE ${RETURN_LIST})
endfunction()


function(avk_release_is_debug_for_imported target)
  if (NOT TARGET ${target})
    message(FATAL_ERROR "Target ${target} does not exist.")
  endif ()

  # Get the existing release import locations
  get_target_property(_implib_release ${target} IMPORTED_IMPLIB_RELEASE)
  get_target_property(_location_release ${target} IMPORTED_LOCATION_RELEASE)

  if (NOT _implib_release AND NOT _location_release)
    message(WARNING "Target ${target} has no IMPORTED_RELEASE properties.")
    return()
  endif ()

  # Mirror release to debug
  if (_implib_release)
    set_target_properties(${target} PROPERTIES IMPORTED_IMPLIB_DEBUG "${_implib_release}")
  endif ()

  if (_location_release)
    set_target_properties(${target} PROPERTIES IMPORTED_LOCATION_DEBUG "${_location_release}")
  endif ()

  message(STATUS "Mapped ${target} release import libs to debug configuration.")
endfunction()


function(avk_create_vulkan_sdk_library_targets)
  if (NOT DEFINED Vulkan_LIBRARY)
    message(FATAL_ERROR "Must use FindVulkan.cmake before calling `avk_create_vulkan_sdk_library_targets`")
  endif ()
  # static volk library
  # if(WIN32)
  #     set(VOLK_NAME "volk.lib")
  # else()
  #     message(FATAL_ERROR "TODO volk")
  # endif()
  # cmake_path(REPLACE_FILENAME Vulkan_LIBRARY "${VOLK_NAME}" OUTPUT_VARIABLE VK_VOLK_LIB_PATH)
  # # we won't set include directory as we assume you are including Vulkan::Vulkan
  # add_library(vk-volk STATIC IMPORTED)
  # set_target_properties(vk-volk PROPERTIES
  #   IMPORTED_LOCATION "${VK_VOLK_LIB_PATH}"
  # )
  # add_library(vk::volk ALIAS vk-volk)
endfunction()
