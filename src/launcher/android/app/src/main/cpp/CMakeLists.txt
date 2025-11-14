# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# create convenience symlink to the currently used libraries
# - avk::core
set(AVK_ANDROID_DEPENDENCIES avk::core)
foreach (elem ${AVK_ANDROID_DEPENDENCIES})
  get_target_property(SRC_DIR ${elem} SOURCE_DIR)
  string(REGEX REPLACE "[:?*/\"\\!']" "_" name ${elem})
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${SRC_DIR} ${CMAKE_SOURCE_DIR}/src/launcher/android/${name}
  )
endforeach ()

# Symlink .clang-tidy and .clang-format from source dir
execute_process(
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/.clang-tidy ${CMAKE_SOURCE_DIR}/src/launcher/android/.clang-tidy
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/.clang-format ${CMAKE_SOURCE_DIR}/src/launcher/android/.clang-format
)

file(GLOB_RECURSE AVK_ANDROID_LAUNCHER_PRIVATE_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cxx"
)

# Creates your game shared library. The name must be the same as the
# one used for loading in your Kotlin/Java or AndroidManifest.txt files.
add_library(avk-android-launcher SHARED)
target_include_directories(avk-android-launcher PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(avk-android-launcher PRIVATE ${AVK_ANDROID_LAUNCHER_PRIVATE_SOURCES})

# Searches for a package provided by the game activity dependency
find_package(game-activity REQUIRED CONFIG)

# Forces the linker to keep the JNI entry point for GameActivity
set(CMAKE_SHARED_LINKER_FLAGS
  "${CMAKE_SHARED_LINKER_FLAGS} -u Java_com_google_androidgamesdk_GameActivity_initializeNativeCode")

if (${AVK_USE_SANITIZERS})
  if (ANDROID_ABI STREQUAL "arm64-v8a")
    # use HWASan
    target_compile_options(avk-android-launcher PRIVATE -fsanitize=hwaddress -fno-omit-frame-pointer)
    target_link_options(avk-android-launcher PRIVATE -fsanitize=hwaddress)
  else ()
    # use ASan
    target_compile_options(avk-android-launcher PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(avk-android-launcher PRIVATE -fsanitize=address)
  endif ()
endif ()

# Configure libraries CMake uses to link your target library.
target_link_libraries(avk-android-launcher
  PRIVATE
  game-activity::game-activity_static
  ${AVK_ANDROID_DEPENDENCIES}
  # android
  # log
)

target_compile_definitions(avk-android-launcher
  PRIVATE
  "AVK_OS_${AVK_OS}" "AVK_COMPILER_${AVK_COMPILER}" "AVK_ARCH_${AVK_ARCH}" "AVK_${AVK_BUILD_FRAGMENT}")

# name must match what you use in LoadLibrary in your kotlin code
set_target_properties(avk-android-launcher PROPERTIES
  OUTPUT_NAME "aethervk"
)

# add validation layers on a debug build
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Android Debug Build: Bundling Vulkan Validation Layers into APK")
  # https://arm-software.github.io/vulkan-sdk/_validation_layer.html
endif ()
