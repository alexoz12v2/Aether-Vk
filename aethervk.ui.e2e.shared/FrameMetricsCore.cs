// Source-linked into aethervk.ui.e2e (Windows), aethervk.ui.e2e.mac and aethervk.ui.e2e.shared.tests.
using System;
using System.Collections.Generic;
using System.Linq;

namespace AetherVk.UI.E2E.Shared;

/// <summary>
/// What a viewport shows, measured off the screen, independent of how the pixels are read: a
/// coarse luminance grid compared frame to frame (did the picture change, and how much), the
/// brightness around a point against the sky in the viewport's corners, and the bright centroid.
/// The Windows <c>FrameMetrics</c> delegates here; the Mac driver feeds it CoreGraphics captures.
/// Luminance is Rec. 709 on the 8-bit sRGB values, 0..1.
/// </summary>
public static class FrameMetricsCore
{
  /// <summary>Grid cells per side of <see cref="Sample"/>.</summary>
  public const int GridSize = 32;

  /// <summary>A cell counts as changed when its luminance moved by more than this (0..1).</summary>
  public const double CellThreshold = 0.01;

  /// <summary>The share of the viewport <see cref="Sample"/> looks at (the nucleus stays centred).</summary>
  public const double CentralShare = 0.9;

  /// <summary>Rec. 709 luminance of one 8-bit pixel.</summary>
  public static double Luma(byte r, byte g, byte b) => (0.0722 * b + 0.7152 * g + 0.2126 * r) / 255.0;

  /// <summary>The viewport's central <paramref name="share"/>, centred (integer pixels).</summary>
  public static E2ERect Central(E2ERect vp, double share = CentralShare)
  {
    int w = (int)(vp.Width * share), h = (int)(vp.Height * share);
    return new E2ERect(vp.X + (vp.Width - w) / 2, vp.Y + (vp.Height - h) / 2, w, h);
  }

  /// <summary>
  /// The n × n cell centres of a full-resolution capture, row by row (the pick the Windows probe
  /// has always made: integer cell centres, no averaging).
  /// </summary>
  public static double[] Grid(double[] full, int width, int height, int n)
  {
    if (full.Length < width * height)
      throw new ArgumentException($"capture has {full.Length} values, not {width}×{height}", nameof(full));
    var lum = new double[n * n];
    for (int j = 0; j < n; j++)
      for (int i = 0; i < n; i++)
        lum[j * n + i] = full[(2 * j + 1) * height / (2 * n) * width + (2 * i + 1) * width / (2 * n)];
    return lum;
  }

  /// <summary>The viewport's central 90 % on a <see cref="GridSize"/>² grid, through the driver's sampler.</summary>
  public static double[] Sample(Func<E2ERect, int, double[]> sampleLuminance, E2ERect vp) =>
    sampleLuminance(Central(vp), GridSize);

  /// <summary>The share of cells whose luminance moved by more than <see cref="CellThreshold"/>.</summary>
  public static double Changed(double[] a, double[] b)
  {
    if (a.Length != b.Length || a.Length == 0)
      throw new ArgumentException($"grids differ in size ({a.Length} vs {b.Length})");
    return a.Zip(b, (x, y) => Math.Abs(x - y) > CellThreshold ? 1.0 : 0.0).Average();
  }

  public static double Mean(double[] values) => values.Length == 0 ? 0 : values.Average();

  /// <summary>The share of values above <paramref name="threshold"/>.</summary>
  public static double LitShare(double[] values, double threshold) =>
    values.Length == 0 ? 0 : values.Count(v => v > threshold) / (double)values.Length;

  /// <summary>The (2·outer+1)² box centred on a pixel.</summary>
  public static E2ERect RingBox(int cx, int cy, int outerPx) =>
    new(cx - outerPx, cy - outerPx, 2 * outerPx + 1, 2 * outerPx + 1);

  /// <summary>The mean of the ring inner..outer (inclusive radii) of a box from <see cref="RingBox"/>.</summary>
  public static double RingMean(double[] box, int innerPx, int outerPx)
  {
    int size = 2 * outerPx + 1;
    double sum = 0;
    int n = 0;
    for (int j = 0; j < size; j++)
      for (int i = 0; i < size; i++)
      {
        int dx = i - outerPx, dy = j - outerPx;
        int r2 = dx * dx + dy * dy;
        if (r2 < innerPx * innerPx || r2 > outerPx * outerPx)
          continue;
        sum += box[j * size + i];
        n++;
      }
    return n == 0 ? 0 : sum / n;
  }

  /// <summary>Four <paramref name="corner"/>² boxes inset <paramref name="inset"/> px into the viewport's corners.</summary>
  public static IReadOnlyList<E2ERect> SkyCorners(E2ERect vp, int corner = 40, int inset = 8)
  {
    int left = vp.X + inset, top = vp.Y + inset;
    int right = vp.X + vp.Width - inset - corner, bottom = vp.Y + vp.Height - inset - corner;
    return [new(left, top, corner, corner), new(right, top, corner, corner),
      new(left, bottom, corner, corner), new(right, bottom, corner, corner)];
  }

  /// <summary>
  /// The mean luminance of the ring <paramref name="innerPx"/>–<paramref name="outerPx"/> around
  /// <paramref name="centre"/> and the sky: the darkest of the four corner boxes (an overlay panel
  /// in one corner cannot raise it). <paramref name="capture"/> returns a rectangle's full-resolution luminance.
  /// </summary>
  public static (double Ring, double Sky) RingAndSky(Func<E2ERect, double[]> capture, E2ERect vp,
    (double X, double Y) centre, int innerPx = 10, int outerPx = 60)
  {
    int cx = (int)Math.Round(centre.X), cy = (int)Math.Round(centre.Y);
    double ring = RingMean(capture(RingBox(cx, cy, outerPx)), innerPx, outerPx);
    double sky = SkyCorners(vp).Min(c => capture(c).Average());
    return (ring, sky);
  }

  /// <summary>
  /// The luminance-weighted centroid of the pixels above <paramref name="threshold"/>, in 0..1 of
  /// the plane (x right, y down); NaN with Count 0 when nothing is that bright.
  /// </summary>
  public static (double X, double Y, int Count) BrightCentroid(double[] lum, int width, int height, double threshold)
  {
    double sx = 0, sy = 0, sw = 0;
    int count = 0;
    for (int y = 0; y < height; y++)
      for (int x = 0; x < width; x++)
      {
        double v = lum[y * width + x];
        if (v <= threshold)
          continue;
        sx += v * (x + 0.5);
        sy += v * (y + 0.5);
        sw += v;
        count++;
      }
    return count == 0 ? (double.NaN, double.NaN, 0) : (sx / sw / width, sy / sw / height, count);
  }

  /// <summary>The larger of the centroid's x and y distances from the centre, in shares of the side.</summary>
  public static double CentreOffset(double x, double y) => Math.Max(Math.Abs(x - 0.5), Math.Abs(y - 0.5));
}
