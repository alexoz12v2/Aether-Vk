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
  public static extern IntPtr GetForegroundWindow();

  [DllImport("user32.dll")]
  public static extern bool IsWindow(IntPtr hWnd);

  /// <summary>An Alt press and release: Windows then lets this process set the foreground window.</summary>
  public static void TapAlt()
  {
    const byte VK_MENU = 0x12;
    const uint KEYEVENTF_KEYUP = 0x0002;
    keybd_event(VK_MENU, 0, 0, UIntPtr.Zero);
    keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, UIntPtr.Zero);
  }

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

  /// <summary>
  /// Luminance (0..1, Rec. 709) of an n × n grid of screen pixels over a rectangle, row by row:
  /// what is on screen (overlays included), without an image library. The cell centres of one
  /// <see cref="CaptureLuminance"/>: a few ms, where a GetPixel per cell stalled on a busy GPU
  /// (~30 s for 32 × 32 while the dust played).
  /// </summary>
  public static double[] SampleLuminance(int left, int top, int width, int height, int n)
  {
    var full = CaptureLuminance(left, top, width, height);
    var lum = new double[n * n];
    for (int j = 0; j < n; j++)
      for (int i = 0; i < n; i++)
        lum[j * n + i] = full[(2 * j + 1) * height / (2 * n) * width + (2 * i + 1) * width / (2 * n)];
    return lum;
  }

  [DllImport("gdi32.dll")]
  private static extern IntPtr CreateCompatibleDC(IntPtr hdc);

  [DllImport("gdi32.dll")]
  private static extern IntPtr CreateCompatibleBitmap(IntPtr hdc, int width, int height);

  [DllImport("gdi32.dll")]
  private static extern IntPtr SelectObject(IntPtr hdc, IntPtr obj);

  [DllImport("gdi32.dll")]
  private static extern bool BitBlt(IntPtr dst, int x, int y, int w, int h, IntPtr src, int sx, int sy, uint rop);

  [DllImport("gdi32.dll")]
  private static extern int GetDIBits(IntPtr hdc, IntPtr bitmap, uint start, uint lines, byte[] bits,
    ref BitmapInfoHeader info, uint usage);

  [DllImport("gdi32.dll")]
  private static extern bool DeleteObject(IntPtr obj);

  [DllImport("gdi32.dll")]
  private static extern bool DeleteDC(IntPtr hdc);

  [StructLayout(LayoutKind.Sequential)]
  private struct BitmapInfoHeader
  {
    public uint Size;
    public int Width, Height;
    public ushort Planes, BitCount;
    public uint Compression, SizeImage;
    public int XPelsPerMeter, YPelsPerMeter;
    public uint ClrUsed, ClrImportant;
    // GetDIBits writes a colour table after the header for some formats; none for 32 bpp BI_RGB
  }

  /// <summary>
  /// Luminance (0..1, Rec. 709) of every screen pixel of a rectangle, row by row from the top:
  /// the full-resolution counterpart of <see cref="SampleLuminance"/> (one BitBlt, overlays included).
  /// </summary>
  public static double[] CaptureLuminance(int left, int top, int width, int height)
  {
    var bits = CaptureBgra(left, top, width, height);
    var lum = new double[width * height];
    for (int k = 0; k < lum.Length; k++)
      lum[k] = (0.0722 * bits[4 * k] + 0.7152 * bits[4 * k + 1] + 0.2126 * bits[4 * k + 2]) / 255.0;
    return lum;
  }

  /// <summary>
  /// A screen rectangle saved as a PNG (one BitBlt, overlays included): milliseconds, where a
  /// WinAppDriver desktop screenshot takes seconds.
  /// </summary>
  public static void SavePng(string path, int left, int top, int width, int height)
  {
    var bgra = CaptureBgra(left, top, width, height);
    // RGB rows, each behind filter byte 0
    var raw = new byte[height * (1 + width * 3)];
    for (int y = 0, o = 0; y < height; y++)
    {
      raw[o++] = 0;
      for (int x = 0; x < width; x++, o += 3)
      {
        int k = 4 * (y * width + x);
        (raw[o], raw[o + 1], raw[o + 2]) = (bgra[k + 2], bgra[k + 1], bgra[k]);
      }
    }
    using var z = new MemoryStream();
    using (var zs = new System.IO.Compression.ZLibStream(z, System.IO.Compression.CompressionLevel.Fastest, leaveOpen: true))
      zs.Write(raw);
    var header = new byte[13];
    System.Buffers.Binary.BinaryPrimitives.WriteInt32BigEndian(header, width);
    System.Buffers.Binary.BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(4), height);
    header[8] = 8; // bit depth
    header[9] = 2; // colour type RGB
    using var f = File.Create(path);
    f.Write([0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    WriteChunk(f, "IHDR", header);
    WriteChunk(f, "IDAT", z.ToArray());
    WriteChunk(f, "IEND", []);
  }

  private static void WriteChunk(Stream f, string type, byte[] data)
  {
    var be = new byte[4];
    System.Buffers.Binary.BinaryPrimitives.WriteInt32BigEndian(be, data.Length);
    f.Write(be);
    var typed = new byte[4 + data.Length];
    System.Text.Encoding.ASCII.GetBytes(type).CopyTo(typed, 0);
    data.CopyTo(typed, 4);
    f.Write(typed);
    System.Buffers.Binary.BinaryPrimitives.WriteUInt32BigEndian(be, System.IO.Hashing.Crc32.HashToUInt32(typed));
    f.Write(be);
  }

  /// <summary>The BGRA bytes of a screen rectangle, rows from the top.</summary>
  private static byte[] CaptureBgra(int left, int top, int width, int height)
  {
    const uint SRCCOPY = 0x00CC0020;
    var screen = GetDC(IntPtr.Zero);
    var mem = CreateCompatibleDC(screen);
    var bmp = CreateCompatibleBitmap(screen, width, height);
    try
    {
      var old = SelectObject(mem, bmp);
      BitBlt(mem, 0, 0, width, height, screen, left, top, SRCCOPY);
      SelectObject(mem, old);
      // negative height: top-down rows; BGRA bytes
      var info = new BitmapInfoHeader
      {
        Size = (uint)Marshal.SizeOf<BitmapInfoHeader>(), Width = width, Height = -height, Planes = 1, BitCount = 32,
      };
      var bits = new byte[width * height * 4];
      if (GetDIBits(mem, bmp, 0, (uint)height, bits, ref info, 0) != height)
        throw new InvalidOperationException("GetDIBits failed");
      return bits;
    }
    finally
    {
      DeleteObject(bmp);
      DeleteDC(mem);
      ReleaseDC(IntPtr.Zero, screen);
    }
  }

  private delegate bool MonitorEnumProc(IntPtr hMonitor, IntPtr hdc, ref Rect rect, IntPtr lParam);

  [DllImport("user32.dll")]
  private static extern bool EnumDisplayMonitors(IntPtr hdc, IntPtr clip, MonitorEnumProc callback, IntPtr lParam);

  [DllImport("user32.dll")]
  private static extern bool ShowWindow(IntPtr hWnd, int cmd);

  [DllImport("user32.dll", SetLastError = true)]
  private static extern bool SetWindowPos(IntPtr hWnd, IntPtr after, int x, int y, int w, int h, uint flags);

  [StructLayout(LayoutKind.Sequential)]
  private struct Rect
  {
    public int Left, Top, Right, Bottom;
  }

  /// <summary>A monitor's bounds, physical desktop px.</summary>
  public sealed record Monitor(int Left, int Top, int Width, int Height);

  /// <summary>Every monitor of the desktop, the primary (the one at the origin) first.</summary>
  public static List<Monitor> Monitors()
  {
    var found = new List<Monitor>();
    EnumDisplayMonitors(IntPtr.Zero, IntPtr.Zero, (IntPtr _, IntPtr _, ref Rect r, IntPtr _) =>
    {
      found.Add(new Monitor(r.Left, r.Top, r.Right - r.Left, r.Bottom - r.Top));
      return true;
    }, IntPtr.Zero);
    return found.OrderBy(m => m.Left != 0 || m.Top != 0).ThenBy(m => m.Left).ToList();
  }

  /// <summary>
  /// Moves a top-level window onto <paramref name="monitor"/> and maximises it there: restored
  /// first (a maximised window cannot be moved), placed inside the monitor, then maximised, which
  /// Windows does on the monitor the window is on.
  /// </summary>
  public static void MaximizeOn(IntPtr hWnd, Monitor monitor)
  {
    const int SW_RESTORE = 9, SW_MAXIMIZE = 3;
    const uint SWP_NOZORDER = 0x0004, SWP_NOACTIVATE = 0x0010;
    ShowWindow(hWnd, SW_RESTORE);
    Thread.Sleep(300);
    if (!SetWindowPos(hWnd, IntPtr.Zero, monitor.Left + 40, monitor.Top + 40, monitor.Width - 80, monitor.Height - 80, SWP_NOZORDER | SWP_NOACTIVATE))
      throw new InvalidOperationException($"SetWindowPos failed ({Marshal.GetLastWin32Error()})");
    Thread.Sleep(300);
    ShowWindow(hWnd, SW_MAXIMIZE);
    Thread.Sleep(500);
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
