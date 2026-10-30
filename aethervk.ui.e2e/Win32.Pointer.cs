using System.Runtime.InteropServices;

namespace AetherVk.UI.E2E;

/// <summary>A pointer drag and the per-monitor scaling, for the billboard placement checks.</summary>
internal static partial class Win32
{
  private const uint MOUSEEVENTF_MOVE = 0x0001;

  [DllImport("user32.dll")]
  private static extern IntPtr MonitorFromPoint(Point pt, uint flags);

  [DllImport("shcore.dll")]
  private static extern int GetDpiForMonitor(IntPtr hMonitor, int dpiType, out uint dpiX, out uint dpiY);

  [StructLayout(LayoutKind.Sequential)]
  private struct Point
  {
    public int X, Y;
  }

  /// <summary>
  /// A left drag as a user makes it: move to the start, press, move in <paramref name="steps"/>
  /// (each followed by a pause so the window gets a pointer-moved event per step), release.
  /// </summary>
  public static void DragFromTo(int x0, int y0, int x1, int y1, int steps = 20)
  {
    SetCursorPos(x0, y0);
    Thread.Sleep(80);
    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, UIntPtr.Zero);
    Thread.Sleep(80);
    for (int i = 1; i <= steps; i++)
    {
      SetCursorPos(x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps);
      mouse_event(MOUSEEVENTF_MOVE, 0, 0, 0, UIntPtr.Zero);
      Thread.Sleep(25);
    }
    Thread.Sleep(80);
    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, UIntPtr.Zero);
    Thread.Sleep(150);
  }

  /// <summary>The scaling (physical px per DIP: 1.0 at 96 dpi, 1.5 at 150 %) of the monitor
  /// showing a desktop point; 1.0 when it cannot be read.</summary>
  public static double ScalingAt(int x, int y)
  {
    const uint MONITOR_DEFAULTTONEAREST = 2;
    const int MDT_EFFECTIVE_DPI = 0;
    var monitor = MonitorFromPoint(new Point { X = x, Y = y }, MONITOR_DEFAULTTONEAREST);
    return monitor != IntPtr.Zero && GetDpiForMonitor(monitor, MDT_EFFECTIVE_DPI, out uint dpiX, out _) == 0 && dpiX > 0
      ? dpiX / 96.0
      : 1.0;
  }

  /// <summary>The scaling of the monitor holding <paramref name="monitor"/>'s centre.</summary>
  public static double ScalingOf(Monitor monitor) =>
    ScalingAt(monitor.Left + monitor.Width / 2, monitor.Top + monitor.Height / 2);
}
