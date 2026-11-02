using System.Drawing;
using System.Runtime.InteropServices;

namespace AetherVk.UI.E2E;

/// <summary>The few extra calls the manual capture needs on top of <c>Win32.cs</c>.</summary>
internal static partial class Win32
{
  [DllImport("user32.dll")]
  private static extern bool GetWindowRect(IntPtr hWnd, out Rect rect);

  [DllImport("user32.dll")]
  private static extern bool PostMessage(IntPtr hWnd, uint message, IntPtr wParam, IntPtr lParam);

  public const byte VK_ESCAPE = 0x1B;

  /// <summary>The BGRA bytes of a screen rectangle, rows from the top (a BitBlt: overlay windows included).</summary>
  public static byte[] Capture(int left, int top, int width, int height) => CaptureBgra(left, top, width, height);

  public static void MoveCursor(int x, int y) => SetCursorPos(x, y);

  /// <summary>A window's visible bounds (the DWM frame, without the invisible resize border), desktop px.</summary>
  public static Rectangle FrameBounds(IntPtr hWnd)
  {
    if (DwmGetWindowAttribute(hWnd, DWMWA_EXTENDED_FRAME_BOUNDS, out var r, Marshal.SizeOf<Rect>()) == 0
        || GetWindowRect(hWnd, out r))
      return Rectangle.FromLTRB(r.Left, r.Top, r.Right, r.Bottom);
    throw new InvalidOperationException("could not read the window bounds");
  }

  /// <summary>Asks a window to close (WM_CLOSE), as its × button does.</summary>
  public static void Close(IntPtr hWnd) => PostMessage(hWnd, 0x0010, IntPtr.Zero, IntPtr.Zero);
}
