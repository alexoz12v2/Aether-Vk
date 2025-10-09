add_subdirectory(core)

if(AVK_OS STREQUAL "WINDOWS")
    add_subdirectory(launcher/windows)
endif()

if(AVK_OS STREQUAL "ANDROID")
    add_subdirectory(launcher/android/app/src/main/cpp)
endif()