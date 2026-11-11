using System.Diagnostics;
using System.Runtime.InteropServices;
using AetherVk.UI.E2E.Shared;

namespace AetherVk.UI.E2E.Mac;

/// <summary>
/// Screen pixels on macOS through CoreGraphics (the counterpart of the Windows BitBlt probe):
/// a rectangle of the main display copied with <c>CGDisplayCreateImageForRect</c> (falling back to
/// <c>CGWindowListCreateImage</c>, then to the <c>screencapture</c> tool when the SDK has dropped
/// the symbols), drawn into an RGBA bitmap. Coordinates in and out are screen pixels; CoreGraphics
/// takes points, converted by the main display's pixel/point ratio (<see cref="Scale"/>).
/// Without Screen Recording permission for the process running <c>dotnet test</c> macOS returns
/// only the desktop wallpaper or black: <c>PixelProbe_MatchesElementScreenshot</c> catches it.
/// Main display only (the suite keeps the app there).
/// </summary>
public static class MacScreen
{
  private const string CoreGraphics = "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics";

  [StructLayout(LayoutKind.Sequential)]
  private struct CGRect(double x, double y, double width, double height)
  {
    public double X = x, Y = y, Width = width, Height = height;
  }

  [DllImport(CoreGraphics)]
  private static extern uint CGMainDisplayID();

  [DllImport(CoreGraphics)]
  private static extern CGRect CGDisplayBounds(uint display);

  /// <summary>The main display's bounds in points (global coordinates, top-left origin).</summary>
  public static (double X, double Y, double Width, double Height) MainBoundsPoints
  {
    get
    {
      var b = CGDisplayBounds(CGMainDisplayID());
      return (b.X, b.Y, b.Width, b.Height);
    }
  }

  [DllImport(CoreGraphics)]
  private static extern IntPtr CGDisplayCopyDisplayMode(uint display);

  [DllImport(CoreGraphics)]
  private static extern nuint CGDisplayModeGetWidth(IntPtr mode);

  [DllImport(CoreGraphics)]
  private static extern nuint CGDisplayModeGetPixelWidth(IntPtr mode);

  [DllImport(CoreGraphics)]
  private static extern void CGDisplayModeRelease(IntPtr mode);

  [DllImport(CoreGraphics)]
  private static extern IntPtr CGDisplayCreateImageForRect(uint display, CGRect rect);

  [DllImport(CoreGraphics)]
  private static extern IntPtr CGWindowListCreateImage(CGRect screenBounds, uint listOption, uint windowId, uint imageOption);

  [DllImport(CoreGraphics)]
  private static extern nuint CGImageGetWidth(IntPtr image);

  [DllImport(CoreGraphics)]
  private static extern nuint CGImageGetHeight(IntPtr image);

  [DllImport(CoreGraphics)]
  private static extern void CGImageRelease(IntPtr image);

  [DllImport(CoreGraphics)]
  private static extern IntPtr CGColorSpaceCreateDeviceRGB();

  [DllImport(CoreGraphics)]
  private static extern void CGColorSpaceRelease(IntPtr space);

  [DllImport(CoreGraphics)]
  private static extern IntPtr CGBitmapContextCreate(IntPtr data, nuint width, nuint height, nuint bitsPerComponent,
    nuint bytesPerRow, IntPtr space, uint bitmapInfo);

  [DllImport(CoreGraphics)]
  private static extern void CGContextDrawImage(IntPtr context, CGRect rect, IntPtr image);

  [DllImport(CoreGraphics)]
  private static extern void CGContextRelease(IntPtr context);

  private const uint kCGWindowListOptionOnScreenOnly = 1;
  private const uint kCGNullWindowID = 0;
  private const uint kCGWindowImageDefault = 0;
  private const uint kCGImageAlphaPremultipliedLast = 1;
  private const uint kCGBitmapByteOrder32Big = 4 << 12;

  private static double? _scale;

  /// <summary>The main display's pixel/point ratio (2 on Retina, also in its scaled modes).</summary>
  public static double Scale
  {
    get
    {
      if (_scale is { } s)
        return s;
      var mode = CGDisplayCopyDisplayMode(CGMainDisplayID());
      if (mode == IntPtr.Zero)
        return 1.0;
      try
      {
        _scale = ScreenScale.FromModes((long)CGDisplayModeGetPixelWidth(mode), (long)CGDisplayModeGetWidth(mode));
      }
      finally
      {
        CGDisplayModeRelease(mode);
      }
      return _scale.Value;
    }
  }

  /// <summary>How the last capture was made (logged by the probe self-test).</summary>
  public static string LastMethod { get; private set; } = "none";

  /// <summary>The rectangle's RGBA pixels, exactly <paramref name="rect"/>'s size (resampled if the capture differs).</summary>
  public static PngImage Capture(E2ERect rect)
  {
    var points = ScreenScale.ToPoints(rect, Scale);
    PngImage? image = null;
    try
    {
      image = CaptureCoreGraphics(points);
    }
    catch (Exception e) when (e is EntryPointNotFoundException or DllNotFoundException)
    {
      LastMethod = $"CoreGraphics unavailable ({e.GetType().Name})";
    }
    image ??= CaptureTool(points);
    return image.Width == rect.Width && image.Height == rect.Height ? image : Resize(image, rect.Width, rect.Height);
  }

  public static double[] CaptureLuminance(E2ERect rect) => Capture(rect).Luminance();

  private static PngImage? CaptureCoreGraphics(PointRect r)
  {
    var cg = new CGRect(r.X, r.Y, r.Width, r.Height);
    var image = CGDisplayCreateImageForRect(CGMainDisplayID(), cg);
    LastMethod = "CGDisplayCreateImageForRect";
    if (image == IntPtr.Zero)
    {
      image = CGWindowListCreateImage(cg, kCGWindowListOptionOnScreenOnly, kCGNullWindowID, kCGWindowImageDefault);
      LastMethod = "CGWindowListCreateImage";
    }
    if (image == IntPtr.Zero)
      return null;
    try
    {
      int w = (int)CGImageGetWidth(image), h = (int)CGImageGetHeight(image);
      if (w <= 0 || h <= 0)
        return null;
      var rgba = new byte[w * h * 4];
      var handle = GCHandle.Alloc(rgba, GCHandleType.Pinned);
      var space = CGColorSpaceCreateDeviceRGB();
      try
      {
        var ctx = CGBitmapContextCreate(handle.AddrOfPinnedObject(), (nuint)w, (nuint)h, 8, (nuint)(w * 4), space,
          kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
        if (ctx == IntPtr.Zero)
          return null;
        // a bitmap context's first memory row is the top of what is drawn: rows come out top-down
        CGContextDrawImage(ctx, new CGRect(0, 0, w, h), image);
        CGContextRelease(ctx);
      }
      finally
      {
        CGColorSpaceRelease(space);
        handle.Free();
      }
      return new PngImage(w, h, rgba);
    }
    finally
    {
      CGImageRelease(image);
    }
  }

  /// <summary><c>screencapture -x -R x,y,w,h</c> (points) to a temporary PNG.</summary>
  private static PngImage CaptureTool(PointRect r)
  {
    var file = Path.Combine(Path.GetTempPath(), $"aethervk-e2e-{Environment.ProcessId}-{Guid.NewGuid():N}.png");
    var psi = new ProcessStartInfo("/usr/sbin/screencapture")
    {
      UseShellExecute = false,
      ArgumentList = { "-x", "-t", "png", "-R", FormattableString.Invariant($"{r.X:0.##},{r.Y:0.##},{r.Width:0.##},{r.Height:0.##}"), file },
    };
    using (var p = Process.Start(psi) ?? throw new InvalidOperationException("screencapture did not start"))
      if (!p.WaitForExit(15_000) || p.ExitCode != 0)
        throw new InvalidOperationException("screencapture failed");
    try
    {
      LastMethod = "screencapture";
      return PngCodec.Read(file);
    }
    finally
    {
      File.Delete(file);
    }
  }

  /// <summary>Nearest-neighbour resize of RGBA pixels.</summary>
  public static PngImage Resize(PngImage src, int width, int height)
  {
    var dst = new byte[width * height * 4];
    for (int y = 0; y < height; y++)
    {
      int sy = Math.Min(src.Height - 1, (int)((y + 0.5) * src.Height / height));
      for (int x = 0; x < width; x++)
      {
        int sx = Math.Min(src.Width - 1, (int)((x + 0.5) * src.Width / width));
        Buffer.BlockCopy(src.Rgba, 4 * (sy * src.Width + sx), dst, 4 * (y * width + x), 4);
      }
    }
    return new PngImage(width, height, dst);
  }
}
