using System;
using System.Collections.Generic;
using AetherVk.Logic.Models;

namespace AetherVk.Logic.Utils;

/// <summary>
/// Finds a comet's nucleus (the photocentre of the inner coma) in an image, in <b>image pixel
/// coordinates</b>: x right, y down, row 0 = the top of the displayed image (texture row order),
/// the centre of pixel (column i, row j) at (i, j). So the image's top-left corner is
/// (−0.5, −0.5) and its centre ((W − 1) / 2, (H − 1) / 2); a FITS pixel (p1, p2) (1-based, rows
/// bottom-up, flipped at import) is at (p1 − 1, H − p2).
/// <para>Two steps: the brightest <see cref="BoxSide"/>² box mean among box centres in the central
/// third of the image (a star elsewhere does not win), then one intensity-weighted centroid of the
/// pixels whose centres lie within <see cref="CentroidRadiusPx"/> of that box centre, each weighted
/// by max(0, value − background), the background being the median of the pixels with
/// <see cref="CentroidRadiusPx"/> &lt; r ≤ <see cref="BackgroundOuterRadiusPx"/>. The centroid is
/// not iterated: a comet's coma and tail are not symmetric, and re-centring the disc on its own
/// centroid walks it down the tail (2 px on the Liverpool 67P stack), whereas one pass from the
/// peak stays on the nucleus.</para>
/// </summary>
public static class NucleusDetector
{
  /// <summary>Side of the box whose mean finds the peak, px (odd: centred on a pixel).</summary>
  public const int BoxSide = 9;

  /// <summary>Radius of the centroid disc, px.</summary>
  public const double CentroidRadiusPx = 12;

  /// <summary>Outer radius of the background annulus (inner: <see cref="CentroidRadiusPx"/>), px.</summary>
  public const double BackgroundOuterRadiusPx = 16;

  /// <summary>The nucleus of <paramref name="image"/> (image pixel coordinates, see the type); null
  /// when the image is too small for the search, blank there, or has no signal above the
  /// background.</summary>
  public static PointD? Detect(IntensityImage image)
  {
    int w = image.Width, h = image.Height;
    var v = image.Values;
    if (w <= 0 || h <= 0 || v.Length < (long)w * h)
      return null;
    if (BrightestBox(v, w, h) is not { } peak)
      return null;
    var (bx, by) = peak;

    double background = Background(v, w, h, bx, by);
    double r2 = CentroidRadiusPx * CentroidRadiusPx;
    int r = (int)Math.Ceiling(CentroidRadiusPx);
    double sw = 0, sx = 0, sy = 0;
    for (int j = Math.Max(0, by - r); j <= Math.Min(h - 1, by + r); j++)
    for (int i = Math.Max(0, bx - r); i <= Math.Min(w - 1, bx + r); i++)
    {
      double dx = i - bx, dy = j - by;
      double value = v[j * w + i];
      if (dx * dx + dy * dy > r2 || double.IsNaN(value))
        continue;
      double weight = value - background;
      if (weight <= 0)
        continue;
      sw += weight;
      sx += weight * i;
      sy += weight * j;
    }
    return sw > 0 ? new PointD(sx / sw, sy / sw) : null;
  }

  /// <summary>Luminance (Rec. 709) of tightly packed RGBA8 texels; a transparent texel is blank.</summary>
  public static IntensityImage FromRgba8(int width, int height, byte[] rgba)
  {
    var values = new float[width * height];
    for (int k = 0; k < values.Length; k++)
    {
      int o = 4 * k;
      values[k] = rgba[o + 3] == 0
        ? float.NaN
        : (float)(0.2126 * rgba[o] + 0.7152 * rgba[o + 1] + 0.0722 * rgba[o + 2]);
    }
    return new IntensityImage(width, height, values);
  }

  /// <summary>
  /// The centre (pixel index) of the brightest <see cref="BoxSide"/>² box mean whose centre lies in
  /// the central third ([W / 3, W − W / 3) × [H / 3, H − H / 3)) and whose box is inside the image;
  /// blank pixels are left out of the mean (a box at least half blank is skipped). The first
  /// (top-left) of equal means wins. Summed-area tables: O(W · H).
  /// </summary>
  private static (int X, int Y)? BrightestBox(float[] v, int w, int h)
  {
    // prefix sums over finite values and their count, (w + 1) × (h + 1)
    var sum = new double[(w + 1) * (h + 1)];
    var count = new int[(w + 1) * (h + 1)];
    for (int j = 0; j < h; j++)
    {
      double rowSum = 0;
      int rowCount = 0;
      for (int i = 0; i < w; i++)
      {
        float x = v[j * w + i];
        if (!float.IsNaN(x))
        {
          rowSum += x;
          rowCount++;
        }
        int k = (j + 1) * (w + 1) + i + 1;
        sum[k] = sum[k - (w + 1)] + rowSum;
        count[k] = count[k - (w + 1)] + rowCount;
      }
    }

    const int half = BoxSide / 2;
    int x0 = Math.Max(half, w / 3), x1 = Math.Min(w - half, w - w / 3);
    int y0 = Math.Max(half, h / 3), y1 = Math.Min(h - half, h - h / 3);
    double best = double.NegativeInfinity;
    (int, int)? at = null;
    for (int y = y0; y < y1; y++)
    for (int x = x0; x < x1; x++)
    {
      int a = (y - half) * (w + 1) + (x - half), b = a + BoxSide, c = a + BoxSide * (w + 1), d = c + BoxSide;
      int n = count[d] - count[b] - count[c] + count[a];
      if (2 * n < BoxSide * BoxSide)
        continue;
      double mean = (sum[d] - sum[b] - sum[c] + sum[a]) / n;
      if (mean > best)
      {
        best = mean;
        at = (x, y);
      }
    }
    return at;
  }

  /// <summary>Median of the finite pixels with <see cref="CentroidRadiusPx"/> &lt; r ≤
  /// <see cref="BackgroundOuterRadiusPx"/> from (cx, cy); 0 when there are none.</summary>
  private static double Background(float[] v, int w, int h, int cx, int cy)
  {
    double rin2 = CentroidRadiusPx * CentroidRadiusPx, rout2 = BackgroundOuterRadiusPx * BackgroundOuterRadiusPx;
    int r = (int)Math.Ceiling(BackgroundOuterRadiusPx);
    var ring = new List<float>();
    for (int j = Math.Max(0, cy - r); j <= Math.Min(h - 1, cy + r); j++)
    for (int i = Math.Max(0, cx - r); i <= Math.Min(w - 1, cx + r); i++)
    {
      double dx = i - cx, dy = j - cy, d2 = dx * dx + dy * dy;
      float x = v[j * w + i];
      if (d2 > rin2 && d2 <= rout2 && !float.IsNaN(x))
        ring.Add(x);
    }
    if (ring.Count == 0)
      return 0;
    ring.Sort();
    int m = ring.Count / 2;
    return ring.Count % 2 == 1 ? ring[m] : 0.5 * (ring[m - 1] + ring[m]);
  }
}
