using System;
using System.Runtime.InteropServices;

namespace AetherVk.Logic.Services;

/// <inheritdoc cref="IPlatformWindowService"/>
public sealed class PlatformWindowService : IPlatformWindowService
{
  /// <inheritdoc/>
  public void SetWindowNonMoveable(nint windowXid)
  {
#if TARGET_IS_LINUX
    if (windowXid == 0) return;

    nint disp = PInvokeX11.XOpenDisplay(0);
    if (disp == 0) return;

    try
    {
      // XA_ATOM = 4 (hardcoded standard predefined atom)
      const nint XA_ATOM      = 4;
      const int  PropModeReplace = 0;

      // _NET_WM_WINDOW_TYPE_DOCK: EWMH dock/panel hint.
      // Compliant WMs (KWin, Mutter) will not allow the user to move or resize a DOCK window
      // via normal WM gestures (Super+LMB, etc.).
      nint typeAtom = PInvokeX11.XInternAtom(disp, "_NET_WM_WINDOW_TYPE", false);
      nint dockAtom = PInvokeX11.XInternAtom(disp, "_NET_WM_WINDOW_TYPE_DOCK", false);

      unsafe
      {
        PInvokeX11.XChangeProperty(disp, windowXid,
          typeAtom, XA_ATOM, 32, PropModeReplace,
          (nint)(&dockAtom), 1);
      }

      // _NET_WM_ALLOWED_ACTIONS = {} (empty list): belt-and-suspenders hint telling the WM
      // that no actions (move, resize, close, etc.) are permitted on this window.
      nint actionsAtom = PInvokeX11.XInternAtom(disp, "_NET_WM_ALLOWED_ACTIONS", false);
      PInvokeX11.XChangeProperty(disp, windowXid,
        actionsAtom, XA_ATOM, 32, PropModeReplace,
        0 /*data=null*/, 0 /*nelements=0*/);

      PInvokeX11.XFlush(disp);
    }
    finally
    {
      PInvokeX11.XCloseDisplay(disp);
    }
#endif
  }
}
