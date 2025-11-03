# Aether-Vk

## Dependencies

- Vulkan SDK >= 1.4 (actually, any version with a slang compiler)
- [Boost library](https://www.boost.org/doc/user-guide/getting-started.html)
  - Possibility of removal if the developer manages to work out Internal Fibers implementation for our supported ABIs

## Build

### Windows

- LLVM required (either downloaded from Visual Studio Installer or standalone)

CMake Configure step

```powershell
cmake .. -DCMAKE_TOOLCHAIN_FILE="..\cmake\Windows.Clang.toolchain.cmake" -G Ninja -DCMAKE_BUILD_TYPE=Debug
```

### Linux

### Android

- Targeted version: Android 11 "R" (API 30) (Minimum to use `GameActivity`)
- Note: We don't support Bazel as a build system for Android, as it is still in early
  stages [Issues Link](https://github.com/bazelbuild/rules_android/issues)

## C++20 Modules Support And C++ Version

C++ version targeted is C++17, mainly because Android doesn't fully reccomend it until `libc++` doesn't
fully develop all C++20 features ([libc++ state](https://libcxx.llvm.org/Status/Cxx20.html))

C++20 Modules are not used because, while Desktop Builds with CMake version >= 3.28 would support it

- Bazel is still WIP

  - [Single Step Compilation PR](https://github.com/bazelbuild/bazel/pull/22553)
  - [Double Step Compilation PR](https://github.com/bazelbuild/bazel/pull/22555)

- Android NDK doesn't support C++20 Modules yet

  - [NDK Issue](https://github.com/android/ndk/issues/1855)

## Known Bugs

- Windows x86_64: first resize scales the content (it shouldn't)

## TO-REMOVE Dependencies

- `Boost::Fiber` (not supported on windows on arm)
- `Boost::Context` is fine! (supported on everything except UWP and Emscripten)
- `ktx`: doesn't work on x86 (32-bit), which is an android ABI (despite it being unusable)
