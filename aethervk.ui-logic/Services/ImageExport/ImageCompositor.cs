using System;
using System.Collections.Generic;
using AetherVk.Logic.Utils;

namespace AetherVk.Logic.Services.ImageExport;

/// <summary>
/// Draws desktop billboards over a captured viewport frame at their on-screen geometry: for every
/// frame pixel under a billboard, the billboard image is sampled bilinearly at the point the
/// pixel's centre maps to (desktop → billboard-local by <see cref="DesktopBillboardGeometry"/>,
/// the same mapping as the gizmo and the windows) and blended with alpha = billboard opacity ×
/// image alpha. A pure function.
/// </summary>
public static class ImageCompositor
{
  /// <summary>
  /// A copy of <paramref name="frameRgba"/> (<paramref name="width"/> × <paramref name="height"/>,
  /// rows from the top, the image of <paramref name="viewport"/>) with
  /// <paramref name="billboards"/> drawn over it in list order.
  /// </summary>
  public static byte[] Composite(
    byte[] frameRgba,
    int width,
    int height,
    ViewportFrame viewport,
    IReadOnlyList<BillboardSnapshot> billboards)
  {
    if (width <= 0 || height <= 0)
      throw new ArgumentOutOfRangeException(nameof(width), "empty frame");
    if (frameRgba.Length < width * height * 4)
      throw new ArgumentException("frame buffer shorter than width × height × 4", nameof(frameRgba));
    var output = new byte[width * height * 4];
    Array.Copy(frameRgba, output, output.Length);
    if (viewport.WidthPx <= 0 || viewport.HeightPx <= 0)
      return output;

    // desktop px per frame px (the frame is the viewport's image: 1 unless the sizes disagree)
    double sx = viewport.WidthPx / width, sy = viewport.HeightPx / height;
    foreach (var snapshot in billboards)
      Draw(output, width, height, viewport, sx, sy, snapshot);
    return output;
  }

  private static void Draw(
    byte[] output, int width, int height, ViewportFrame viewport, double sx, double sy, BillboardSnapshot snapshot)
  {
    var b = snapshot.Billboard;
    int iw = snapshot.ImageWidth, ih = snapshot.ImageHeight;
    if (iw <= 0 || ih <= 0 || snapshot.Rgba.Length < iw * ih * 4 || b.WidthPx <= 0 || b.HeightPx <= 0)
      return;
    double opacity = Math.Max(0.0, Math.Min(1.0, b.Alpha));
    if (opacity <= 0)
      return;

    // the frame pixels the rotated billboard can touch: its corners' bounding box
    double hw = b.WidthPx / 2.0, hh = b.HeightPx / 2.0;
    double minX = double.PositiveInfinity, minY = double.PositiveInfinity;
    double maxX = double.NegativeInfinity, maxY = double.NegativeInfinity;
    foreach (var corner in new[] { new PointD(-hw, -hh), new PointD(hw, -hh), new PointD(hw, hh), new PointD(-hw, hh) })
    {
      var d = DesktopBillboardGeometry.ToDesktop(b, corner);
      double fx = (d.X - viewport.Origin.X) / sx, fy = (d.Y - viewport.Origin.Y) / sy;
      minX = Math.Min(minX, fx);
      maxX = Math.Max(maxX, fx);
      minY = Math.Min(minY, fy);
      maxY = Math.Max(maxY, fy);
    }
    int px0 = Math.Max(0, (int)Math.Floor(minX)), px1 = Math.Min(width - 1, (int)Math.Ceiling(maxX));
    int py0 = Math.Max(0, (int)Math.Floor(minY)), py1 = Math.Min(height - 1, (int)Math.Ceiling(maxY));
    if (px0 > px1 || py0 > py1)
      return;

    var rgba = snapshot.Rgba;
    for (int py = py0; py <= py1; py++)
    {
      for (int px = px0; px <= px1; px++)
      {
        // the pixel's centre on the desktop, then in the billboard's unrotated frame
        var desktop = new PointD(viewport.Origin.X + (px + 0.5) * sx, viewport.Origin.Y + (py + 0.5) * sy);
        var local = DesktopBillboardGeometry.ToLocal(b, desktop);
        if (local.X < -hw || local.X > hw || local.Y < -hh || local.Y > hh)
          continue;
        // image pixel coordinates with pixel centres at integers: the image fills the billboard
        double u = (local.X / b.WidthPx + 0.5) * iw - 0.5;
        double v = (local.Y / b.HeightPx + 0.5) * ih - 0.5;
        Sample(rgba, iw, ih, u, v, out double r, out double g, out double bl, out double a);
        double alpha = opacity * a / 255.0;
        if (alpha <= 0)
          continue;
        int k = 4 * (py * width + px);
        output[k] = Blend(output[k], r, alpha);
        output[k + 1] = Blend(output[k + 1], g, alpha);
        output[k + 2] = Blend(output[k + 2], bl, alpha);
        output[k + 3] = 255;
      }
    }
  }

  private static byte Blend(byte under, double over, double alpha) =>
    (byte)Math.Max(0, Math.Min(255, Math.Round(under * (1 - alpha) + over * alpha)));

  /// <summary>Bilinear sample of the RGBA image at (<paramref name="u"/>, <paramref name="v"/>),
  /// pixel centres at integers, edges clamped.</summary>
  private static void Sample(byte[] rgba, int iw, int ih, double u, double v,
    out double r, out double g, out double b, out double a)
  {
    int x0 = (int)Math.Floor(u), y0 = (int)Math.Floor(v);
    double fx = u - x0, fy = v - y0;
    int xa = Clamp(x0, iw), xb = Clamp(x0 + 1, iw);
    int ya = Clamp(y0, ih), yb = Clamp(y0 + 1, ih);
    double w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
    int k00 = 4 * (ya * iw + xa), k10 = 4 * (ya * iw + xb), k01 = 4 * (yb * iw + xa), k11 = 4 * (yb * iw + xb);
    r = rgba[k00] * w00 + rgba[k10] * w10 + rgba[k01] * w01 + rgba[k11] * w11;
    g = rgba[k00 + 1] * w00 + rgba[k10 + 1] * w10 + rgba[k01 + 1] * w01 + rgba[k11 + 1] * w11;
    b = rgba[k00 + 2] * w00 + rgba[k10 + 2] * w10 + rgba[k01 + 2] * w01 + rgba[k11 + 2] * w11;
    a = rgba[k00 + 3] * w00 + rgba[k10 + 3] * w10 + rgba[k01 + 3] * w01 + rgba[k11 + 3] * w11;
  }

  private static int Clamp(int i, int n) => i < 0 ? 0 : i >= n ? n - 1 : i;
}
