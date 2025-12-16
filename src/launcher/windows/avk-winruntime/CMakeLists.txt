file(GLOB_RECURSE AVK_WINRUNTIME_FILES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(avk-winruntime SHARED)
target_sources(avk-winruntime PRIVATE
  ${AVK_WINRUNTIME_FILES}
)
set(DEPS "")
if (CMAKE_BUILD_TYPE STREQUAL Debug AND AVK_USE_SANITIZERS)
  list(APPEND DEPS win-asan-dynamic)
endif ()
list(LENGTH DEPS DEPS_LENGTH)

if (DEPS_LENGTH GREATER 0)
  target_link_libraries(avk-winruntime PRIVATE win-asan-dynamic)
endif ()
target_include_directories(avk-winruntime PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(avk-winruntime
  PRIVATE
  "AVK_WINRUNTIME_EXPORTS"
  "AVK_OS_${AVK_OS}" "AVK_COMPILER_${AVK_COMPILER}" "AVK_ARCH_${AVK_ARCH}" "AVK_${AVK_BUILD_FRAGMENT}"
  "_UNICODE" "UNICODE" "WIN32_LEAN_AND_MEAN" "NOMINMAX")

target_compile_options(avk-winruntime PRIVATE ${AVK_CXX_TARGET_COMPILE_FLAGS})
# Produce deterministic binaries
target_link_options(avk-winruntime PRIVATE -Wl,/Brepro)

# windows ignores SOVERSION, but whatever
set_target_properties(avk-winruntime PROPERTIES
  OUTPUT_NAME "AetherVk.Runtime" # DLL is the default OUTPUT_SUFFIX on windows for SHARED
  VERSION 1.0.0
  SOVERSION 1
)

# copy dependent DLLs into the build tree
if (DEPS_LENGTH GREATER 0)
  add_custom_command(TARGET avk-winruntime POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:avk-winruntime> $<TARGET_FILE_DIR:avk-winruntime>
    COMMAND_EXPAND_LISTS
    USES_TERMINAL
  )
endif ()

# code signing (we don't care)
# add_custom_command(TARGET avk-winruntime POST_BUILD
#   COMMAND signtool sign
#     /fd SHA256
#     /tr http://timestamp.digicert.com # I made this up
#     /td SHA256
#     /a $<TARGET_FILE:avk-winruntime>
# )

####################################################################################
# CMake Install -> nuget package production
# Directory structure:
# AetherVk.Runtime.nuget/
# ├── runtimes/
# │   └── win-x64/
# │       └── native/
# │           ├── Aethervk.Runtime.dll
# │           └── Aethervk.Runtime.pdb
# │
# ├── build/
# │   └── native/
# │       └── AetherVk.Runtime.targets
# │
# └── AetherVk.Runtime.nuspec
####################################################################################

# The install prefix must be set when configuring CMake, e.g.:
#   cmake -S . -B <build> -G Ninja -DCMAKE_INSTALL_PREFIX=build/install -DCMAKE_INSTALL_CONFIG=Debug
#   cmake --install <build> --component <COMP_NAME>
# Example:
#  Y:\Aether-Vk> cmake --install cmake-build-debug-system --prefix .\src\launcher\windows\AetherVk\AetherVk.Runtime --config Debug --component nuget_runtime

# TODO copy PDB only in debug builds/install
# TODO support for Windows on ARM (64)

## install commands are executed for each source directory in the order they are declared.
## that's because cmake progressively generates a cmake_install.cmake script for each source subdirectory
## each statement is guarded with a component equality check with CMAKE_INSTALL_COMPONENT, which you can
## - give to the `cmake --install` through the `--component` flag
## - specify to a install command through `COMPONENT`

# First, at installation time, starting from CMAKE_INSTALL_PREFIX and CMAKE_INSTALL_CONFIG_NAME, create the required
# directory structure for a nuget package
# TODO windows on ARM
# TODO clean if exists
install(CODE [[
  file(MAKE_DIRECTORY
    "${CMAKE_INSTALL_PREFIX}/nuget/${CMAKE_INSTALL_CONFIG_NAME}/AetherVk.Runtime.nuget/runtimes/win-x64/native"
    "${CMAKE_INSTALL_PREFIX}/nuget/${CMAKE_INSTALL_CONFIG_NAME}/AetherVk.Runtime.nuget/build/native"
  )
]]
  COMPONENT nuget_runtime
)
install(TARGETS avk-winruntime
  # escape variable: ugly trick to force install time variable expansion (tested on windows x64 cmake 4.1)
  RUNTIME DESTINATION "nuget/\${CMAKE_INSTALL_CONFIG_NAME}/AetherVk.Runtime.nuget/runtimes/win-x64/native"
  COMPONENT nuget_runtime
)
# copy dependent dlls
install(FILES $<TARGET_RUNTIME_DLLS:avk-winruntime>
  DESTINATION "nuget/\${CMAKE_INSTALL_CONFIG_NAME}/AetherVk.Runtime.nuget/runtimes/win-x64/native"
  COMPONENT nuget_runtime
)
install(FILES $<TARGET_PDB_FILE:avk-winruntime>
  DESTINATION "nuget/\${CMAKE_INSTALL_CONFIG_NAME}/AetherVk.Runtime.nuget/runtimes/win-x64/native"
  COMPONENT nuget_runtime
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/AetherVk.Runtime.nuspec"
  DESTINATION "nuget/\${CMAKE_INSTALL_CONFIG_NAME}/AetherVk.Runtime.nuget"
  COMPONENT nuget_runtime
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/AetherVk.Runtime.targets"
  DESTINATION "nuget/\${CMAKE_INSTALL_CONFIG_NAME}/AetherVk.Runtime.nuget/build/native"
  COMPONENT nuget_runtime
)
# TODO insert code to locate the nuget executable, don't rely on the PATH
install(CODE [[
  execute_process(
    COMMAND nuget pack AetherVk.Runtime.nuspec
    WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}/nuget/${CMAKE_INSTALL_CONFIG_NAME}/AetherVk.Runtime.nuget"
    COMMAND_ERROR_IS_FATAL ANY
    COMMAND_ECHO STDOUT
  )
]]
  COMPONENT nuget_runtime
)
