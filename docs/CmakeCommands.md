# Notes on cmake commands

## some commands

Windows, configure/generation step inside build directory with Ninja generator

```sh
cmake .. -DCMAKE_TOOLCHAIN_FILE="..\cmake\Windows.Clang.toolchain.cmake" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DAVK_USE_SANITIZERS=ON
```
