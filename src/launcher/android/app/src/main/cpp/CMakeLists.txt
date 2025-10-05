# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Creates your game shared library. The name must be the same as the
# one used for loading in your Kotlin/Java or AndroidManifest.txt files.
add_library(avk-android-launcher SHARED
        main.cpp
        AndroidOut.cpp
        Renderer.cpp
        Shader.cpp
        TextureAsset.cpp
        Utility.cpp)

# Searches for a package provided by the game activity dependency
find_package(game-activity REQUIRED CONFIG)

# Forces the linker to keep the JNI entry point for GameActivity
set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -u Java_com_google_androidgamesdk_GameActivity_initializeNativeCode")

# Configure libraries CMake uses to link your target library.
target_link_libraries(avk-android-launcher
        # The game activity
        game-activity::game-activity_static

        # EGL and other dependent libraries required for drawing
        # and interacting with Android system
        EGL
        GLESv3
        jnigraphics
        android
        log)

target_compile_definitions(avk-android-launcher
  PRIVATE 
    "AVK_OS_${AVK_OS}" "AVK_COMPILER_${AVK_COMPILER}" "AVK_ARCH_${AVK_ARCH}")