# Aether-Vk

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

## C++20 Modules Support And C++ Version

C++ version targeted is C++17, mainly because Android doesn't fully reccomend it until `libc++` doesn't
fully develop all C++20 features ([libc++ state](https://libcxx.llvm.org/Status/Cxx20.html))

C++20 Modules are not used because, while Desktop Builds with CMake version >= 3.28 would support it

- Bazel is still WIP

  - [Single Step Compilation PR](https://github.com/bazelbuild/bazel/pull/22553)
  - [Double Step Compilation PR](https://github.com/bazelbuild/bazel/pull/22555)

- Android NDK doesn't support C++20 Modules yet

  - [NDK Issue](https://github.com/android/ndk/issues/1855)
