using System.Runtime.InteropServices;
using System.Text;

namespace AetherVk.UI.E2E;

/// <summary>The few user32 calls WinAppDriver has no command for (wheel scroll, window lookup).</summary>
internal static partial class Win32
{
  private delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

  [DllImport("user32.dll")]
  private static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lParam);

  [DllImport("user32.dll")]
  private static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);

  [DllImport("user32.dll")]
  private static extern bool IsWindowVisible(IntPtr hWnd);

  [DllImport("user32.dll", CharSet = CharSet.Unicode)]
  private static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int maxCount);

  [DllImport("user32.dll", CharSet = CharSet.Unicode)]
  private static extern int GetClassName(IntPtr hWnd, StringBuilder name, int maxCount);

  [DllImport("user32.dll")]
  public static extern bool SetForegroundWindow(IntPtr hWnd);

  [DllImport("user32.dll")]
  private static extern bool SetCursorPos(int x, int y);

  [DllImport("user32.dll")]
  private static extern void keybd_event(byte vk, byte scan, uint flags, UIntPtr extraInfo);

  [DllImport("user32.dll")]
  private static extern void mouse_event(uint flags, int dx, int dy, int data, UIntPtr extraInfo);

  private const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
  private const uint MOUSEEVENTF_LEFTUP = 0x0004;
  private const uint MOUSEEVENTF_WHEEL = 0x0800;
  private const int WHEEL_DELTA = 120;

  public sealed record Window(IntPtr Handle, string Title, string ClassName);

  /// <summary>Visible top-level windows of a process.</summary>
  public static List<Window> TopLevelWindows(int processId)
  {
    var found = new List<Window>();
    EnumWindows((hWnd, _) =>
    {
      GetWindowThreadProcessId(hWnd, out uint pid);
      if (pid == processId && IsWindowVisible(hWnd))
      {
        var title = new StringBuilder(512);
        GetWindowText(hWnd, title, title.Capacity);
        var cls = new StringBuilder(256);
        GetClassName(hWnd, cls, cls.Capacity);
        found.Add(new Window(hWnd, title.ToString(), cls.ToString()));
      }
      return true;
    }, IntPtr.Zero);
    return found;
  }

  /// <summary>Scrolls the wheel at a screen point (positive = towards the user, i.e. down).</summary>
  public static void WheelAt(int x, int y, int notchesDown)
  {
    SetCursorPos(x, y);
    Thread.Sleep(30);
    mouse_event(MOUSEEVENTF_WHEEL, 0, 0, -notchesDown * WHEEL_DELTA, UIntPtr.Zero);
    Thread.Sleep(120);
  }

  /// <summary>
  /// A left click as a user makes it: move, settle, press, hold, release. Avalonia misses the
  /// instant down/up WinAppDriver injects for an element click.
  /// </summary>
  public static void ClickAt(int x, int y)
  {
    SetCursorPos(x, y);
    Thread.Sleep(60);
    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, UIntPtr.Zero);
    Thread.Sleep(60);
    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, UIntPtr.Zero);
    Thread.Sleep(120);
  }

  /// <summary>
  /// A key press to whatever window has keyboard focus (WinAppDriver's keys may refocus the
  /// session's top-level window first, away from the viewport's native child).
  /// </summary>
  public static void PressKey(byte virtualKey)
  {
    const uint KEYEVENTF_KEYUP = 0x0002;
    keybd_event(virtualKey, 0, 0, UIntPtr.Zero);
    Thread.Sleep(50);
    keybd_event(virtualKey, 0, KEYEVENTF_KEYUP, UIntPtr.Zero);
    Thread.Sleep(150);
  }

  [DllImport("user32.dll")]
  private static extern IntPtr GetDC(IntPtr hWnd);

  [DllImport("user32.dll")]
  private static extern int ReleaseDC(IntPtr hWnd, IntPtr hdc);

  [DllImport("gdi32.dll")]
  private static extern uint GetPixel(IntPtr hdc, int x, int y);

  /// <summary>
  /// Luminance (0..1, Rec. 709) of an n × n grid of screen pixels over a rectangle, row by row:
  /// what is on screen (overlays included), without an image library.
  /// </summary>
  public static double[] SampleLuminance(int left, int top, int width, int height, int n)
  {
    var dc = GetDC(IntPtr.Zero);
    try
    {
      var lum = new double[n * n];
      for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
        {
          uint c = GetPixel(dc, left + (2 * i + 1) * width / (2 * n), top + (2 * j + 1) * height / (2 * n));
          lum[j * n + i] = (0.2126 * (c & 0xFF) + 0.7152 * ((c >> 8) & 0xFF) + 0.0722 * ((c >> 16) & 0xFF)) / 255.0;
        }
      return lum;
    }
    finally
    {
      ReleaseDC(IntPtr.Zero, dc);
    }
  }

  public const byte VK_RETURN = 0x0D;
  public const byte VK_CONTROL = 0x11;
  public const byte VK_MENU = 0x12;

  /// <summary>A modifier held around a key press, e.g. (VK_MENU, 'D') for Alt+D.</summary>
  public static void Chord(byte modifier, byte virtualKey)
  {
    const uint KEYEVENTF_KEYUP = 0x0002;
    keybd_event(modifier, 0, 0, UIntPtr.Zero);
    Thread.Sleep(50);
    PressKey(virtualKey);
    keybd_event(modifier, 0, KEYEVENTF_KEYUP, UIntPtr.Zero);
    Thread.Sleep(150);
  }

  [DllImport("user32.dll", SetLastError = true)]
  private static extern bool OpenClipboard(IntPtr owner);

  [DllImport("user32.dll")]
  private static extern bool EmptyClipboard();

  [DllImport("user32.dll", SetLastError = true)]
  private static extern IntPtr SetClipboardData(uint format, IntPtr data);

  [DllImport("user32.dll")]
  private static extern bool CloseClipboard();

  [DllImport("kernel32.dll", SetLastError = true)]
  private static extern IntPtr GlobalAlloc(uint flags, UIntPtr bytes);

  [DllImport("kernel32.dll")]
  private static extern IntPtr GlobalLock(IntPtr mem);

  [DllImport("kernel32.dll")]
  private static extern bool GlobalUnlock(IntPtr mem);

  /// <summary>
  /// Pastes text into the focused control through the clipboard: typed keys follow the active
  /// layout (it-IT turns WinAppDriver's backslash into '#'), a paste is the text as is.
  /// </summary>
  public static void Paste(string text)
  {
    const uint CF_UNICODETEXT = 13, GMEM_MOVEABLE = 0x0002;
    for (int attempt = 0; !OpenClipboard(IntPtr.Zero); attempt++)
    {
      if (attempt == 20)
        throw new InvalidOperationException("the clipboard stayed busy");
      Thread.Sleep(50);
    }
    try
    {
      EmptyClipboard();
      var chars = (text + '\0').ToCharArray();
      var mem = GlobalAlloc(GMEM_MOVEABLE, (UIntPtr)(chars.Length * sizeof(char)));
      Marshal.Copy(chars, 0, GlobalLock(mem), chars.Length);
      GlobalUnlock(mem);
      if (SetClipboardData(CF_UNICODETEXT, mem) == IntPtr.Zero)
        throw new InvalidOperationException($"SetClipboardData failed ({Marshal.GetLastWin32Error()})");
    }
    finally
    {
      CloseClipboard();
    }
    Chord(VK_CONTROL, (byte)'V');
  }
}
