// Source-linked into aethervk.ui.e2e (Windows), aethervk.ui.e2e.mac and aethervk.ui.e2e.shared.tests.
using System;

namespace AetherVk.UI.E2E.Shared;

/// <summary>
/// Points ↔ pixels. The Mac2 driver reports element frames in points; the scenarios work in screen
/// pixels (the exported PNG and the swapchain are pixel sized), so every rectangle crosses this
/// once. A pixel rectangle covers every pixel the point rectangle touches.
/// </summary>
public static class ScreenScale
{
  private const double Eps = 1e-6;

  /// <summary>The display's pixel/point ratio from its mode (e.g. 2880 px over 1440 pt → 2).</summary>
  public static double FromModes(long pixelWidth, long pointWidth) =>
    pixelWidth > 0 && pointWidth > 0 ? (double)pixelWidth / pointWidth : 1.0;

  public static E2ERect ToPixels(PointRect r, double scale)
  {
    int left = (int)Math.Floor(r.X * scale + Eps);
    int top = (int)Math.Floor(r.Y * scale + Eps);
    int right = (int)Math.Ceiling((r.X + r.Width) * scale - Eps);
    int bottom = (int)Math.Ceiling((r.Y + r.Height) * scale - Eps);
    return new E2ERect(left, top, Math.Max(0, right - left), Math.Max(0, bottom - top));
  }

  public static PointRect ToPoints(E2ERect r, double scale) =>
    new(r.X / scale, r.Y / scale, r.Width / scale, r.Height / scale);

  public static (int X, int Y) PointToPixel(double x, double y, double scale) =>
    ((int)Math.Round(x * scale), (int)Math.Round(y * scale));

  public static (double X, double Y) PixelToPoint(double x, double y, double scale) => (x / scale, y / scale);

  /// <summary>
  /// Nearest-neighbour resample of a row-major <paramref name="sw"/> × <paramref name="sh"/> plane
  /// to <paramref name="dw"/> × <paramref name="dh"/> (a capture taken at another scale than asked).
  /// </summary>
  public static double[] Resample(double[] src, int sw, int sh, int dw, int dh)
  {
    if (src.Length != sw * sh)
      throw new ArgumentException($"plane has {src.Length} values, not {sw}×{sh}", nameof(src));
    if (sw == dw && sh == dh)
      return (double[])src.Clone();
    var dst = new double[dw * dh];
    for (int y = 0; y < dh; y++)
    {
      int sy = Math.Min(sh - 1, (int)((y + 0.5) * sh / dh));
      for (int x = 0; x < dw; x++)
      {
        int sx = Math.Min(sw - 1, (int)((x + 0.5) * sw / dw));
        dst[y * dw + x] = src[sy * sw + sx];
      }
    }
    return dst;
  }
}
