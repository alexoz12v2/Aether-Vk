using System.Runtime.InteropServices;
using Avalonia;
using Avalonia.Controls;

namespace AetherVk.Interop;

public static class MouseUtils
{
  [DllImport("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")]
  private static extern void CGWarpMouseCursorPosition(CGPoint newCursorPosition);

  private struct CGPoint
  {
    public double X;
    public double Y;
  }

  [DllImport("user32.dll")]
  [return: MarshalAs(UnmanagedType.Bool)]
  private static extern bool SetCursorPos(int X, int Y);

  public static void SetCursorPosition(PixelPoint screenPt, double renderScaling)
  {
    if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
    {
      // macOS uses logical coordinates for CGWarpMouseCursorPosition
      CGWarpMouseCursorPosition(
        new CGPoint { X = screenPt.X / renderScaling, Y = screenPt.Y / renderScaling }
      );
    }
    else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
    {
      // Windows SetCursorPos uses physical pixels
      SetCursorPos(screenPt.X, screenPt.Y);
    }
    // X11 / Wayland warping is complex and omitted for now, assuming primarily macOS/Windows usage.
  }
}
