# Notes on Compilation

## Compile Command Example

```sh
C:\PROGRA~1\MICROS~1\2022\COMMUN~1\VC\Tools\Llvm\x64\bin\CLANG_~1.EXE 
  -DAVK_ARCH_X86_64 
  -DAVK_COMPILER_CLANG 
  -DAVK_OS_WINDOWS 
  -DNOMINMAX 
  -DUNICODE 
  -DWIN32_LEAN_AND_MEAN 
  -isystem "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/ATLMFC/include" 
  -isystem "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/include" 
  -isystem "C:/Program Files (x86)/Windows Kits/10/include/10.0.22621.0/ucrt" 
  -isystem "C:/Program Files (x86)/Windows Kits/10/include/10.0.22621.0/shared" 
  -isystem "C:/Program Files (x86)/Windows Kits/10/include/10.0.22621.0/um" 
  -isystem "C:/Program Files (x86)/Windows Kits/10/include/10.0.22621.0/winrt" 
  -isystem "C:/Program Files (x86)/Windows Kits/10/include/10.0.22621.0/cppwinrt"
  -fvisibility=hidden 
  -fno-fast-math 
  -faddrsig 
  -fstrict-aliasing 
  -nogpuinc -nogpulib 
  -stdlib=libc++ 
  -fstack-protector-all 
  -march=x86-64-v3 
  -fsanitize=address -fsanitize=undefined 
  -flto -fsanitize=cfi 
  -g -O0 
  -std=gnu++17 
  -D_DEBUG 
  -D_DLL -D_MT 
  -Xclang --dependent-lib=msvcrtd 
  -Xclang -gcodeview 
  -Wall -Wextra -pedantic -Werror 

  -o src\\launcher\\windows\\CMakeFiles\\avk-windows-launcher.dir\\main.cpp.obj 
  -c Y:\\Aether-Vk\\src\\launcher\\windows\\main.cpp
```

## Tools in repo

slangc, bazel

```sh
bazel run //:slangc -- <arguments>
```

slangc, cmake (after the configure step, assume directory used called `cmake-build`)

```sh
./cmake-build/vulkan-sdk/slangc <arguments>
```
