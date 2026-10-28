namespace AetherVk.UI.E2E;

/// <summary>
/// Ridges of a dust fan by position angle, on a north-up / east-left luminance image: PA counts
/// from up (north) towards left (east), PA = atan2(−dx, −dy) with x right and y down.
/// <para>
/// Each ring [lo, hi) km about the nucleus is cut into 3° bins; a bin's value is its median
/// luminance divided by the median of the bin medians, so a ridge is a relative excess. This ring
/// normalisation is the same on the render and on the observation, which is high-pass processed
/// (its large-scale background subtracted): only relative structure is compared, never levels.
/// The same algorithm as the session tool fanpa.py, which measured the 2021-12-27 JPEG.
/// </para>
/// </summary>
internal static class FanRidges
{
  public const int BinDeg = 3;
  private const int Bins = 360 / BinDeg;
  // 5-bin (15°) running mean before peak picking, and the least relative excess counted as a ridge
  private const int SmoothBins = 5;
  public const double MinExcess = 1.03;

  /// <summary>Position angle (deg, [0, 360)) of an image offset (x right, y down) from the nucleus.</summary>
  public static double PositionAngle(double dx, double dy) => (Math.Atan2(-dx, -dy) * 180.0 / Math.PI + 360.0) % 360.0;

  /// <summary>
  /// The smoothed ring profile (Bins values, bin b centred on PA (b + ½)·3°), or null when the ring
  /// has fewer than 200 pixels inside the image.
  /// </summary>
  public static double[]? Profile(double[] lum, int width, int height, double cx, double cy, double kmPerPx,
    double loKm, double hiKm)
  {
    var perBin = new List<double>[Bins];
    for (int b = 0; b < Bins; b++)
      perBin[b] = new List<double>();
    int count = 0;
    double hiPx = hiKm / kmPerPx, loPx = loKm / kmPerPx;
    int y0 = Math.Max(0, (int)(cy - hiPx)), y1 = Math.Min(height - 1, (int)(cy + hiPx) + 1);
    int x0 = Math.Max(0, (int)(cx - hiPx)), x1 = Math.Min(width - 1, (int)(cx + hiPx) + 1);
    for (int y = y0; y <= y1; y++)
      for (int x = x0; x <= x1; x++)
      {
        double dx = x - cx, dy = y - cy, r = Math.Sqrt(dx * dx + dy * dy);
        if (r < loPx || r >= hiPx)
          continue;
        perBin[(int)(PositionAngle(dx, dy) / BinDeg) % Bins].Add(lum[y * width + x]);
        count++;
      }
    if (count < 200)
      return null;
    var med = perBin.Select(l => l.Count > 0 ? Median(l) : double.NaN).ToArray();
    double ringMedian = Median(med.Where(v => !double.IsNaN(v)).ToList());
    if (ringMedian <= 0)
      ringMedian = 1.0 / 255.0; // a black ring: any light is an excess
    var rel = med.Select(v => double.IsNaN(v) ? 1.0 : v / ringMedian).ToArray();
    var smooth = new double[Bins];
    for (int b = 0; b < Bins; b++)
    {
      double sum = 0;
      for (int k = -SmoothBins / 2; k <= SmoothBins / 2; k++)
        sum += rel[(b + k + Bins) % Bins];
      smooth[b] = sum / SmoothBins;
    }
    return smooth;
  }

  /// <summary>
  /// Local maxima above <see cref="MinExcess"/>, strongest first, as (PA, excess). A flat top (the
  /// smoothing widens a narrow ridge into one) is one peak at its middle.
  /// </summary>
  public static List<(double Pa, double Excess)> Peaks(double[] profile)
  {
    const double eps = 1e-9;
    var peaks = new List<(double, double)>();
    for (int b = 0; b < Bins; b++)
    {
      double v = profile[b];
      // b starts a plateau: strictly above its left neighbour
      if (v <= MinExcess || v <= profile[(b - 1 + Bins) % Bins] + eps)
        continue;
      int end = b;
      while (end - b < Bins - 1 && Math.Abs(profile[(end + 1) % Bins] - v) <= eps)
        end++;
      if (profile[(end + 1) % Bins] < v - eps)
        peaks.Add(((((b + end) / 2.0 + 0.5) * BinDeg) % 360.0, v));
    }
    return peaks.OrderByDescending(p => p.Item2).ToList();
  }

  /// <summary>
  /// Whether the ring is clipped: bins within 2 % of the maximum span at least
  /// <paramref name="minSpanDeg"/>. A flat top hides the ridges (their peaks are noise on it); only
  /// the fan's edges are measurable there.
  /// </summary>
  public static bool IsSaturated(double[] profile, double minSpanDeg = 30)
  {
    double max = profile.Max();
    return profile.Count(v => v >= 0.98 * max) * BinDeg >= minSpanDeg;
  }

  /// <summary>
  /// The fan's edges: the PAs (counter-clockwise from <paramref name="Lo"/> to
  /// <paramref name="Hi"/>) where the contiguous run of bins around the maximum drops below half
  /// its excess, 1 + (max − 1)/2, each at the outermost bin still above it.
  /// </summary>
  public static (double Lo, double Hi) Edges(double[] profile)
  {
    int top = Array.IndexOf(profile, profile.Max());
    double half = 1 + (profile[top] - 1) / 2;
    int lo = top, hi = top;
    while (hi - top < Bins - 1 && profile[(hi + 1) % Bins] >= half)
      hi++;
    while (top - lo < Bins - 1 && profile[(lo - 1 + Bins) % Bins] >= half)
      lo--;
    return ((((lo + Bins) % Bins) + 0.5) * BinDeg, ((hi % Bins) + 0.5) * BinDeg);
  }

  /// <summary>Largest relative excess with its PA inside [fromPa, toPa] (clockwise-free: fromPa ≤ toPa).</summary>
  public static double MaxIn(double[] profile, double fromPa, double toPa)
  {
    double best = 0;
    for (int b = 0; b < Bins; b++)
    {
      double pa = (b + 0.5) * BinDeg;
      if (pa >= fromPa && pa <= toPa)
        best = Math.Max(best, profile[b]);
    }
    return best;
  }

  /// <summary>Smallest angular distance (deg, 0..180) between two PAs.</summary>
  public static double AngleBetween(double a, double b)
  {
    double d = Math.Abs(a - b) % 360.0;
    return d > 180.0 ? 360.0 - d : d;
  }

  /// <summary>The peak nearest to <paramref name="pa"/> among the given ones, or null.</summary>
  public static (double Pa, double Excess)? Nearest(IEnumerable<(double Pa, double Excess)> peaks, double pa) =>
    peaks.Select(p => ((double Pa, double Excess)?)p).MinBy(p => AngleBetween(p!.Value.Pa, pa));

  public static string Describe(double[] profile) =>
    string.Join(", ", Peaks(profile).Take(4).Select(p => $"{p.Pa:F0}°×{p.Excess:F2}"));

  private static double Median(List<double> v)
  {
    v.Sort();
    int n = v.Count;
    return n % 2 == 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
  }
}
