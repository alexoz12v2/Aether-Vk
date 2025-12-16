
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AetherVk.Core.PInvoke
{
    public static partial class AvkRuntime
    {
        [LibraryImport("AetherVk.Runtime.dll", EntryPoint = "avkSomeFunc", StringMarshalling = StringMarshalling.Utf16)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static unsafe partial void SomeFunc();
    }
}