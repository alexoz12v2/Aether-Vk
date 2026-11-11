using System;
using System.Runtime.InteropServices;

namespace AetherVk.Logic.Services;

/// <summary>
/// The one Xlib connection every Vulkan surface of the process is created on (Linux), main viewport
/// and nucleus mini-view alike.
/// <para>
/// The NVIDIA X11 Vulkan driver crashes in <c>vkQueuePresentKHR</c> when two surfaces of one process
/// were created on different <c>Display*</c> connections (the X11 viewport tests reproduce it with
/// each view's input-handler connection and with a freshly opened one; a shared connection works).
/// So no surface uses its input handler's connection: the handlers keep their own connections for
/// their event loops, and this one never carries an event loop.
/// </para>
/// <para>
/// Opened lazily with <c>XOpenDisplay(NULL)</c> (after the <c>XInitThreads</c> of Program.cs, which
/// makes it safe from the render thread) and never closed: it lives as long as the process, so no
/// surface or swapchain can outlive it, whatever the order viewports and handlers are disposed in.
/// </para>
/// </summary>
public static class X11SurfaceDisplay
{
#if TARGET_IS_LINUX
  private static readonly object s_gate = new();
  private static nint s_display;
#endif

  /// <summary>The shared <c>Display*</c>, opened on the first call; 0 off Linux or when no X server
  /// is reachable (the next call tries again).</summary>
  public static nint Get()
  {
#if TARGET_IS_LINUX
    if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
      return 0;
    lock (s_gate)
    {
      if (s_display == 0)
      {
        s_display = PInvokeX11.XOpenDisplay(0);
        if (s_display == 0)
          Console.WriteLine("[X11SurfaceDisplay] XOpenDisplay failed: no DISPLAY or X server");
      }
      return s_display;
    }
#else
    return 0;
#endif
  }
}
