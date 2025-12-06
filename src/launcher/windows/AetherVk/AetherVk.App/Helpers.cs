
using Microsoft.UI.Xaml;
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AetherVk.App.Helpers;

// example of winproc🆗: https://dotmorten.github.io/WinUIEx/concepts/WindowMessageMonitor.html

internal static class SystemError
{
    public const UInt32 ERROR_SUCCESS = 0;
}

internal static class WindowStyle
{
    // values from https://learn.microsoft.com/en-us/windows/win32/winmsg/extended-window-styles
    public const Int32 GWL_EXSTYLE = -20;
    public const Int32 GWL_WNDPROC = -4;

    public const Int32 WS_EX_TOOLWINDOW = 0x00000080;
    public const Int32 WS_EX_CONTROLPARENT = 0x00010000;
    public const Int32 WS_EX_APPWINDOW = 0x00040000;

    // Helper methods (true => No failure)
    public static bool HideFromTaskbar(this Window window) // extension
    {
        IntPtr hWnd = WinRT.Interop.WindowNative.GetWindowHandle(window);
        IntPtr exStyle = User32.GetWindowLongPtr(hWnd, GWL_EXSTYLE);
        if (Marshal.GetLastPInvokeError() != SystemError.ERROR_SUCCESS)
        {
            return false;
        }

        // remove app window, add tool window
        exStyle &= ~WS_EX_APPWINDOW;
        exStyle |= WS_EX_TOOLWINDOW;

        User32.SetWindowLongPtr(hWnd, GWL_EXSTYLE, exStyle);
        return Marshal.GetLastPInvokeError() == SystemError.ERROR_SUCCESS;
    }
}

// https://learn.microsoft.com/en-us/dotnet/standard/native-interop/
internal static partial class User32
{
    [LibraryImport("user32.dll", EntryPoint = "GetWindowLongPtrW", StringMarshalling = StringMarshalling.Utf16, SetLastError = true)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvStdcall) })]
    public static unsafe partial IntPtr GetWindowLongPtr(IntPtr hWnd, Int32 nIndex);

    [LibraryImport("user32.dll", EntryPoint = "SetWindowLongPtrW", StringMarshalling = StringMarshalling.Utf16, SetLastError = true)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvStdcall) })]
    public static unsafe partial IntPtr SetWindowLongPtr(IntPtr hWnd, Int32 nIndex, IntPtr dwNewLong);
}
