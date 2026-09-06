using System;
using System.Runtime.InteropServices;

namespace AetherVk.Logic.Services;

/// <inheritdoc cref="IPlatformWindowService"/>
public sealed class PlatformWindowService : IPlatformWindowService
{
  /// <inheritdoc/>
  public void SetWindowAsCompanionPanel(nint windowHandle)
  {
#if TARGET_IS_LINUX
    if (windowHandle == 0)
      return;

    nint disp = PInvokeX11.XOpenDisplay(0);
    if (disp == 0)
      return;

    try
    {
      // XA_ATOM = 4 (hardcoded standard predefined atom)
      const nint XA_ATOM = 4;
      const int PropModeReplace = 0;

      // _NET_WM_WINDOW_TYPE_UTILITY: companion panel hint.
      //   - No title bar or WM decorations (WMs honour SystemDecorations=None too, but
      //     UTILITY reinforces it for WMs that check the type before the MWM hints).
      //   - Not listed in the taskbar / pager.
      //   - Does NOT carry the implicit "above all windows" semantic that DOCK has.
      //     DOCK panels are globally always-on-top by EWMH convention; UTILITY panels
      //     stack normally and respect _NET_WM_STATE_ABOVE only when it is explicitly set.
      nint typeAtom = PInvokeX11.XInternAtom(disp, "_NET_WM_WINDOW_TYPE", false);
      nint utilityAtom = PInvokeX11.XInternAtom(disp, "_NET_WM_WINDOW_TYPE_UTILITY", false);

      unsafe
      {
        PInvokeX11.XChangeProperty(
          disp,
          windowHandle,
          typeAtom,
          XA_ATOM,
          32,
          PropModeReplace,
          (nint)(&utilityAtom),
          1
        );
      }

      // _NET_WM_ALLOWED_ACTIONS = {} (empty list): belt-and-suspenders hint telling the WM
      // that no actions (move, resize, close, etc.) are permitted on this window.
      nint actionsAtom = PInvokeX11.XInternAtom(disp, "_NET_WM_ALLOWED_ACTIONS", false);
      PInvokeX11.XChangeProperty(
        disp,
        windowHandle,
        actionsAtom,
        XA_ATOM,
        32,
        PropModeReplace,
        0 /*data=null*/
        ,
        0 /*nelements=0*/
      );

      PInvokeX11.XFlush(disp);
    }
    finally
    {
      PInvokeX11.XCloseDisplay(disp);
    }
#endif
    // Windows/macOS: no-op.
    // ShowInTaskbar=False + SystemDecorations=None in AXAML, combined with the ownership
    // relationship established by Avalonia's Show(ownerWindow), handle all companion-panel
    // semantics at the framework level without any OS PInvoke.
  }

  /// <inheritdoc/>
  public nint GetRootWindowHandle()
  {
#if TARGET_IS_LINUX
    nint disp = PInvokeX11.XOpenDisplay(0);
    if (disp == 0)
      return 0;
    try
    {
      return PInvokeX11.XDefaultRootWindow(disp);
    }
    finally
    {
      PInvokeX11.XCloseDisplay(disp);
    }
#else
    return 0; // Windows/macOS: ignored by the no-op SetOverlayAbove.
#endif
  }

  /// <inheritdoc/>
  public void SetOverlayAbove(nint overlayHandle, nint rootHandle, bool raise)
  {
#if TARGET_IS_LINUX
    if (overlayHandle == 0 || rootHandle == 0)
      return;

    nint disp = PInvokeX11.XOpenDisplay(0);
    if (disp == 0)
      return;

    try
    {
      // EWMH §5.3 — send a _NET_WM_STATE ClientMessage to the root window to add or remove
      // _NET_WM_STATE_ABOVE on the overlay window.
      //   data[0] = 1 (_NET_WM_STATE_ADD)    when raise=true
      //   data[0] = 0 (_NET_WM_STATE_REMOVE)  when raise=false
      //   data[1] = _NET_WM_STATE_ABOVE atom
      //   data[2] = 0   (second property — none)
      //   data[3] = 1   (source indication: normal application)
      nint wmStateAtom = PInvokeX11.XInternAtom(disp, "_NET_WM_STATE", false);
      nint aboveAtom = PInvokeX11.XInternAtom(disp, "_NET_WM_STATE_ABOVE", false);

      unsafe
      {
        PInvokeX11.XEvent ev = default;
        // XClientMessageEvent.data is a union containing `long l[5]`.
        // CRITICAL (LP64 trap): When format=32, Xlib expects the data array to be C `long`s.
        // On 64-bit Linux (LP64), `sizeof(long)` is 64 bits (8 bytes), NOT 32 bits.
        // In C#, `int` is always 32 bits. We must cast to `nint*` so each element consumes
        // 8 bytes, avoiding data corruption where the X server reads two 32-bit values as one.
        ev.xclient.type = XEventName.ClientMessage;
        ev.xclient.window = overlayHandle;
        ev.xclient.message_type = wmStateAtom;
        ev.xclient.format = 32;

        PInvokeX11.XClientMessageEvent* xclientPtr = &ev.xclient;
        nint* dataLongs = (nint*)xclientPtr->data;
        dataLongs[0] = raise ? 1 : 0; // _NET_WM_STATE_ADD or _NET_WM_STATE_REMOVE
        dataLongs[1] = aboveAtom; // first property: _NET_WM_STATE_ABOVE
        dataLongs[2] = 0; // second property: none
        dataLongs[3] = 1; // source indication: normal application
        dataLongs[4] = 0; // safety padding

        PInvokeX11.XSendEvent(
          disp,
          rootHandle,
          propagate: 0,
          eventMask: XEventMask.SubstructureNotifyMask | XEventMask.SubstructureRedirectMask,
          &ev
        );
      }

      PInvokeX11.XFlush(disp);
    }
    finally
    {
      PInvokeX11.XCloseDisplay(disp);
    }
#endif
    // Windows/macOS: no-op.
    // Win32: owned windows (set via Show(ownerWindow) → SetWindowLongPtr GWL_HWNDPARENT)
    //        stay above their owner automatically without HWND_TOPMOST.
    // macOS: child windows (addChildWindow:ordered:NSWindowAbove, set by Avalonia Show(owner))
    //        always render above their parent and move with it.
    // In both cases, the OS handles the correct Z-order relative to the owner and other apps
    // without any explicit action on focus change.
  }

  /// <inheritdoc/>
  public void SetOverlayOverrideRedirect(nint overlayHandle)
  {
#if TARGET_IS_LINUX
    if (overlayHandle == 0)
      return;

    nint disp = PInvokeX11.XOpenDisplay(0);
    if (disp == 0)
      return;
    try
    {
      // CWOverrideRedirect = 1 << 9 = 0x200  (unsigned long valuemask → nuint)
      // Setting override_redirect = 1 tells Xwayland to stop asking the WM (Mutter) to manage
      // this window. Xwayland then exports it as an independent Wayland surface, which Mutter
      // composites using full GPU alpha-blending AFTER the Vulkan sub-surface — exactly the
      // correct compositing order. This bypasses both the top-level Z-order policy (which put
      // the MainWindow above the overlay) and the child-window backing-store merge (which
      // punched transparent holes into the MainWindow).
      // Xlib Bool is int (4 bytes on LP64), NOT C# bool (1 byte) — use 1/0 as int.
      var attrs = new PInvokeX11.XSetWindowAttributes { override_redirect = 1 };
      PInvokeX11.XChangeWindowAttributes(disp, overlayHandle, 0x200, ref attrs);
      PInvokeX11.XFlush(disp);
    }
    finally
    {
      PInvokeX11.XCloseDisplay(disp);
    }
#endif
    // Windows/macOS: no-op — Show(ownerWindow) already guarantees correct Z-order.
  }

  /// <inheritdoc/>
  public void SetCursorPosition(int x, int y)
  {
#if TARGET_IS_LINUX
    nint disp = PInvokeX11.XOpenDisplay(0);
    if (disp != 0)
    {
      try
      {
        nint rootWindow = PInvokeX11.XDefaultRootWindow(disp);
        PInvokeX11.XWarpPointer(disp, 0, rootWindow, 0, 0, 0, 0, x, y);
        PInvokeX11.XFlush(disp);
      }
      finally
      {
        PInvokeX11.XCloseDisplay(disp);
      }
    }
#elif TARGET_IS_WINDOWS
    Windows.Win32.PInvoke.SetCursorPos(x, y);
#elif TARGET_IS_OSX
    PInvokeCoreGraphics.CGWarpMouseCursorPosition(new PInvokeObjC.CGPoint { X = x, Y = y });
#endif
  }
}
