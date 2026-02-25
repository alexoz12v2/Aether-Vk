# Using WinUI3 with XAML Islands on a Vulkan Win32 C++ application

## Creation of the runtime

Assume we have an executable which links against the `/SUBSYSTEM:WINDOWS`, and creates a
`HWND` primary window through the classic Win32 API.

We'd like to achieve the following

- Support modern Windows Frameworks for Application Development
- Have the control to assign to a child `HWND` to a `VkSwapchain` and still be linked to 
  input events properly

To support that, we'd like to include in the project usage of the `C++/WinRT` library, which
allows us to access `WinUI 3`.

The main issue is that WinUI was though to be used with `MSBuild` on a Visual Studio solution,
while we'd like to be independent of the build toolchain as much as possible

## Copy include files inside the binary directory

the `cppwinrt.exe` command line tool allows us to copy header files for the `C++/WinRT` library

```txt
winrt/Microsoft.UI.Xaml.h
winrt/Windows.Foundation.h
winrt/Windows.UI.h
...
```

With the following command

```shell
cppwinrt -input sdk -reference sdk -output $OutPath
```

## Notes

The `App.xaml` for XAML Islands is minimal, as it does **Not** create a window. It only
initializes the XAML framework so controls can be created

WinUI 3 XAML Islands do not use `xamlc` to auto generate `.g.h` files like WPF or UWP.

Instead you write a XAML file and its matching C++ class files, while the C++/WinRT
parses at runtime the XAML data and the XAML data controls is loaded at runtime 
with `Application.LoadComponent()`

Therefore, a cmake script needs to


- Generate the required WinRT projection headers -> generate a pre-compiled header
- Copy XAML files in the output folder, such that the application can access them at runtime

## The Grand Conclusion

C++/WinRT is fine, but XAML on C++/WinRT isn't, as it is only supported on a Visual Studio
build platform.

Therefore, we turn towards alternative solutions

1. If we insist on keeping each OS UI Separate
   - The windows solution becomes WinUI 3 with C# .NET
2. Solution which tries to keep UI Cross Platform
   - Pure Vulkan Solution: Dear ImGui
   - Managed Language Solution: Dart + Flutter

## Exploration of a C# based solution


