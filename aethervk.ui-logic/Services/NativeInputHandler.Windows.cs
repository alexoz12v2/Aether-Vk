using System;
#if TARGET_IS_WINDOWS
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AetherVk.Logic.Services.NativeInput;
#endif

// CsWin32 generated
#if TARGET_IS_WINDOWS
using Windows.Win32;
using Windows.Win32.Foundation;
using Windows.Win32.Graphics.Gdi;
using Windows.Win32.UI.Input.KeyboardAndMouse;
using Windows.Win32.UI.WindowsAndMessaging;
#endif

namespace AetherVk.Logic.Services;

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

// C#12 feature
using unsafe WNDPROC = delegate* unmanaged[Stdcall]<HWND, uint, WPARAM, LPARAM, LRESULT>;

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
      _oldWndProc = (WNDPROC)PInvoke.SetWindowLong((HWND)_handle, WINDOW_LONG_PTR_INDEX.GWL_WNDPROC, (int)(nint)wndProcPtr);

    if (_traceLevel >= TraceLevel.Basic)
      Log(TraceLevel.Basic, $"[Win32] Subclassed HWND {_handle:X}");

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
      // --- Temporary Background Rendering
      if (msg == PInvoke.WM_ERASEBKGND && !(instance._bgBrush != HBRUSH.Null))
      {
        PInvoke.GetClientRect(hWnd, out RECT rect);
        PInvoke.FillRect(new HDC((nint)wParam.Value), &rect, instance._bgBrush);
      }

      // --- Input Interception ---
      InterceptInputMessage(instance, hWnd, msg, wParam, lParam);

      // TODO logging of most important messages related to keyboard (raw input api) and mouse
      // (still raw input api I think)

      // Forward the original window procedure so standard OS behaviour (focus, resizing) persists
      if ((int)instance._oldWndProc != 0)
        return PInvoke.CallWindowProc(instance._oldWndProc, hWnd, msg, wParam, lParam);
    }

    return PInvoke.DefWindowProc(hWnd, msg, wParam, lParam);
  }

  /// <summary>
  /// To be called during a window procedure. Handles input using pre-packaged standard message eg
  /// `WM_KEYDOWN`, `WM_MOUSEMOVE` instead of the Raw Input API. Why?
  /// - Coordinate System -> Raw input gives you hardware delta and ignores mouse acceleration and
  ///   bounds. Instead with this we can get Absolute pointer position relative to client area of
  ///   this window
  /// - Keyboard abstraction -> Raw input gives you hardware scan code, hence you'd need to
  ///   translate it manually given the keyboard layout. Standard messages push keys through
  ///   `TranslateMessage` automatically.
  /// - Consistency with MacOS implementation
  /// </summary>
  private static void InterceptInputMessage(WindowsNativeInputHandler instance, HWND hWnd, uint msg, WPARAM wParam, LPARAM lParam)
  {
    // Windows Vritual Key masks passed in wParam during mouse messages
    // https://learn.microsoft.com/en-us/windows/win32/inputdev/wm-lbuttondown
    const nuint MK_LBUTTON = 0x0001;
    const nuint MK_MBUTTON = 0x0010;
    const nuint MK_RBUTTON = 0x0002;

    switch (msg)
    {
      // --- KEYBOARD --
      case PInvoke.WM_KEYDOWN:
      case PInvoke.WM_SYSKEYDOWN:
        instance.PublishKeyEvent((uint)wParam.Value, isDown: true, GetModifiers());
        break;

      case PInvoke.WM_KEYUP:
      case PInvoke.WM_SYSKEYUP:
        instance.PublishKeyEvent((uint)wParam.Value, isDown: false, GetModifiers());
        break;

      // --- MOUSE MOVEMENT --
      case PInvoke.WM_MOUSEMOVE:
        // Only publish motion when a button is held (drag). Hover-only motion
        // is not consumed by any camera mode — consistent with Linux/macOS.
        if ((wParam.Value & MK_LBUTTON) != 0)
          instance.PublishMouseEvent(GetX(lParam), GetY(lParam), MouseButton.Left,   isDown: true, GetModifiers());
        if ((wParam.Value & MK_RBUTTON) != 0)
          instance.PublishMouseEvent(GetX(lParam), GetY(lParam), MouseButton.Right,  isDown: true, GetModifiers());
        if ((wParam.Value & MK_MBUTTON) != 0)
          instance.PublishMouseEvent(GetX(lParam), GetY(lParam), MouseButton.Middle, isDown: true, GetModifiers());
        break;

      // --- MOUSE CLICKS --
      case PInvoke.WM_LBUTTONDOWN:
        PInvoke.SetCapture(hWnd); // Mimick AppKit: Lock input to this window for the out-of-bounds drags
        instance.PublishMouseEvent(GetX(lParam), GetY(lParam), MouseButton.Left, isDown: true, GetModifiers());
        break;
      case PInvoke.WM_LBUTTONUP:
        instance.PublishMouseEvent(GetX(lParam), GetY(lParam), MouseButton.Left, isDown: false, GetModifiers());
        if ((wParam.Value & (MK_LBUTTON | MK_MBUTTON | MK_RBUTTON)) == 0)
          PInvoke.ReleaseCapture();

        break;

      case PInvoke.WM_RBUTTONDOWN:
        PInvoke.SetCapture(hWnd); // Mimick AppKit: Lock input to this window for the out-of-bounds drags
        instance.PublishMouseEvent(GetX(lParam), GetY(lParam), MouseButton.Right, isDown: true, GetModifiers());
        break;
      case PInvoke.WM_RBUTTONUP:
        instance.PublishMouseEvent(GetX(lParam), GetY(lParam), MouseButton.Right, isDown: false, GetModifiers());
        if ((wParam.Value & (MK_LBUTTON | MK_MBUTTON | MK_RBUTTON)) == 0)
          PInvoke.ReleaseCapture();

        break;

      case PInvoke.WM_MBUTTONDOWN:
        PInvoke.SetCapture(hWnd); // Mimick AppKit: Lock input to this window for the out-of-bounds drags
        instance.PublishMouseEvent(GetX(lParam), GetY(lParam), MouseButton.Middle, isDown: true, GetModifiers());
        break;
      case PInvoke.WM_MBUTTONUP:
        instance.PublishMouseEvent(GetX(lParam), GetY(lParam), MouseButton.Middle, isDown: false, GetModifiers());
        if ((wParam.Value & (MK_LBUTTON | MK_MBUTTON | MK_RBUTTON)) == 0)
          PInvoke.ReleaseCapture();

        break;
    }
  }

  #region Win32 Macro Equivalents

  // Reconstruct C/C++ GET_X_LPARAM and GET_Y_LPARAM macros
  // natively, Windows uses Top-left (0,0). No inversion is needed unlike MacOS
  // Masking to 0xFFFF and casting to `short` is important to preserve negative coordinates
  // (Two's Complement) in the event the user clicks inside and drags the mouse outside the client
  // bounds
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static double GetX(LPARAM lParam) => (short)(lParam.Value & 0xFFFF);

  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  private static double GetY(LPARAM lParam) => (short)((lParam.Value >> 16) & 0xFFFF);

  /// <summary>
  /// Reads the state of modifier keys at the time hte message was added to the message queue.
  /// GetKeyState() is tied to the thread message queue, completely avoiding race conditions
  /// </summary>
  /// <seealso href="https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getkeystate" />
  private static NativeModifierFlags GetModifiers()
  {
    NativeModifierFlags flags = NativeModifierFlags.None;
    // High order bit (0x8000) is 1 if key down
    // Note: We are using `GetKeyState` and not `GetAsyncKeyState`, cause the former checks the
    // state at the moment the message arrived, while the latter checks the state right now, which
    // would desynchronize the input struct.
    if ((PInvoke.GetKeyState((int)VIRTUAL_KEY.VK_SHIFT) & 0x8000) != 0)
      flags |= NativeModifierFlags.Shift;
    if ((PInvoke.GetKeyState((int)VIRTUAL_KEY.VK_CONTROL) & 0x8000) != 0)
      flags |= NativeModifierFlags.Control;
    if ((PInvoke.GetKeyState((int)VIRTUAL_KEY.VK_MENU) & 0x8000) != 0)
      flags |= NativeModifierFlags.Alt;
    if (((PInvoke.GetKeyState((int)VIRTUAL_KEY.VK_LWIN) & 0x8000) != 0) ||
        ((PInvoke.GetKeyState((int)VIRTUAL_KEY.VK_RWIN) & 0x8000) != 0))
      flags |= NativeModifierFlags.Super;

    return flags;
  }

  #endregion
}

#endif
