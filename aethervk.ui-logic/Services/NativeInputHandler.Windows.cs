using System;
#if TARGET_IS_WINDOWS
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
#endif

// CsWin32 generated
#if TARGET_IS_WINDOWS
using Windows.Win32;
using Windows.Win32.Foundation;
using Windows.Win32.Graphics.Gdi;
using Windows.Win32.UI.WindowsAndMessaging;
#endif

namespace AetherVk.Logic.Services;

// C#12 feature
#if TARGET_IS_WINDOWS
using unsafe WNDPROC = delegate* unmanaged[Stdcall]<HWND, uint, WPARAM, LPARAM, LRESULT>;
#endif

#if !TARGET_IS_WINDOWS

public unsafe class WindowsNativeInputHandler(IntPtr handle, string handleDescriptor, TraceLevel traceLevel, IUiThreadDispatcher dispatcher, ISchedulerProvider schedulerProvider)
  : NativeInputHandlerBase(handle, handleDescriptor, traceLevel, dispatcher, schedulerProvider)
{
  protected override bool HookEvents()
  {
    return false;
  }

  protected override void UnhookEvents()
  {
  }

  protected override void DoSetSolidColor(byte r, byte g, byte b)
  {
  }
}

#else

public unsafe class WindowsNativeInputHandler(IntPtr handle, string handleDescriptor, TraceLevel traceLevel, IUiThreadDispatcher dispatcher, ISchedulerProvider schedulerProvider)
    : NativeInputHandlerBase(handle, handleDescriptor, traceLevel, dispatcher, schedulerProvider)
{

  /// <summary>AOT Safe instance mapping for static unmanaged callbacks</summary>
  private static readonly ConcurrentDictionary<IntPtr, WindowsNativeInputHandler> s_instances = [];

  private WNDPROC _oldWndProc;
  private HBRUSH _bgBrush = HBRUSH.Null;

  protected override bool HookEvents()
  {
    s_instances[_handle] = this;
    WNDPROC wndProcPtr = &WndProcHook;

    if (IntPtr.Size == 8)
      _oldWndProc = (WNDPROC)PInvoke.SetWindowLongPtr((HWND)_handle, WINDOW_LONG_PTR_INDEX.GWL_WNDPROC, (nint)wndProcPtr);
    else
      _oldWndProc = (WNDPROC)PInvoke.SetWindowLong((HWND)_handle, WINDOW_LONG_PTR_INDEX.GWL_WNDPROC, (int)wndProcPtr);

    return false;
  }

  protected override void UnhookEvents()
  {
    if (IntPtr.Size == 8)
      PInvoke.SetWindowLongPtr((HWND)_handle, WINDOW_LONG_PTR_INDEX.GWL_WNDPROC, (nint)_oldWndProc);
    else
      PInvoke.SetWindowLong((HWND)_handle, WINDOW_LONG_PTR_INDEX.GWL_WNDPROC, (int)_oldWndProc);

    s_instances.TryRemove(_handle, out _);
  }

  protected override void DoSetSolidColor(byte r, byte g, byte b)
  {
    if (_bgBrush != HBRUSH.Null) PInvoke.DeleteObject(_bgBrush);
    _bgBrush = PInvoke.CreateSolidBrush(new COLORREF(r | ((uint)g << 8) | ((uint)b << 16)));

    PInvoke.InvalidateRect((HWND)_handle, (RECT*)null, new BOOL(1));
  }

  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvStdcall)])]
  private static LRESULT WndProcHook(HWND hWnd, uint msg, WPARAM wParam, LPARAM lParam)
  {
    if (s_instances.TryGetValue(hWnd, out var instance))
    {
      if (msg == PInvoke.WM_ERASEBKGND && !(instance._bgBrush != HBRUSH.Null))
      {
        PInvoke.GetClientRect(hWnd, out RECT rect);
        PInvoke.FillRect(new HDC((nint)wParam.Value), &rect, instance._bgBrush);
      }

      // TODO logging of most important messages related to keyboard (raw input api) and mouse
      // (still raw input api I think)

      if ((int)instance._oldWndProc != 0)
        return PInvoke.CallWindowProc(instance._oldWndProc, hWnd, msg, wParam, lParam);
    }

    return PInvoke.DefWindowProc(hWnd, msg, wParam, lParam);
  }
}

#endif
