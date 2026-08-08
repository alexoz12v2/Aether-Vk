
namespace AetherVk.Logic { }

// Stuff which was introduced in later versions of .NET, brought here to .NET Standard 2.0
// We integrated `System.Memory` and `System.Runtime.CompilerServices.Unsafe` so that we can have a
// .NET Standard 2.0 unmanaged/native API, with minimal overhead and AOT compatibility
//
// 1. System.Memory -> ReadOnlySpan<byte> enables the usage of C# 11 UTF-8 string literals "..."u8,
//    and the compiler will put then in the DLL's .rdata section
//    Such bytes can be passed directly to macOS/Objective-C as C-strings without allocations
//
// 2. System.Runtime.CompilerServices.Unsafe: use `Unsafe.SkipInit` to bypass the `initobj` IL
//    instruction, which wastes cycles zero-filling structs such as Win32's `RECT`
//
// 3. WHile we don't have [LibraryImport] in .NET Standard 2.0, we can use [DllImport] with only
//    blittable arguments as documented https://learn.microsoft.com/en-us/dotnet/standard/native-interop/disabled-marshalling
//    to make this work without any IL marshalling stubs

#if NETSTANDARD2_0
#pragma warning disable IDE0130 // Namespace does not match folder structure

namespace System.Runtime.InteropServices
{
  [AttributeUsage(AttributeTargets.Method, Inherited = false)]
  public sealed class UnmanagedCallersOnlyAttribute : Attribute
  {
    public Type[]? CallConvs;
    public string? EntryPoint;
  }
}

// records (apparently already there)
// namespace System.Runtime.CompilerServices
// {
//   internal static class IsExternalInit { }
// }

#pragma warning restore IDE0130
#endif
