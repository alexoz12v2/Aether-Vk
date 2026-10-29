using System;
using System.Collections.Generic;
using System.Globalization;

namespace AetherVk.Logic.Utils;

/// <summary>One plotted sample (data units, not pixels).</summary>
public readonly struct PlotPoint
{
  public PlotPoint(double x, double y)
  {
    X = x;
    Y = y;
  }

  public double X { get; }
  public double Y { get; }

  public override string ToString() => $"({X.ToString("G6", CultureInfo.InvariantCulture)}, {Y.ToString("G6", CultureInfo.InvariantCulture)})";
}

/// <summary>An axis tick: its position in data units and its label.</summary>
public readonly struct PlotTick
{
  public PlotTick(double value, string label)
  {
    Value = value;
    Label = label;
  }

  public double Value { get; }
  public string Label { get; }
}

/// <summary>A polyline of the plot. <see cref="StyleIndex"/> 0 = primary colour, 1 = secondary (dashed).</summary>
public sealed class PlotSeries
{
  public PlotSeries(string label, IReadOnlyList<PlotPoint> points, int styleIndex = 0)
  {
    Label = label;
    Points = points;
    StyleIndex = styleIndex;
  }

  public string Label { get; }
  public IReadOnlyList<PlotPoint> Points { get; }
  public int StyleIndex { get; }
}

/// <summary>
/// Everything a curve plot draws, computed in the logic layer so it is unit-testable: the series,
/// the axis ranges (log or linear), the ticks and an optional vertical marker. On a log axis a
/// value ≤ 0 is drawn on the bottom edge (Q_d beyond the cutoff).
/// </summary>
public sealed class CurvePlotModel
{
  public IReadOnlyList<PlotSeries> Series { get; set; } = Array.Empty<PlotSeries>();
  public double XMin { get; set; }
  public double XMax { get; set; }
  public double YMin { get; set; }
  public double YMax { get; set; }
  public bool XLog { get; set; }
  public bool YLog { get; set; }
  public string XLabel { get; set; } = "";
  public string YLabel { get; set; } = "";
  public IReadOnlyList<PlotTick> XTicks { get; set; } = Array.Empty<PlotTick>();
  public IReadOnlyList<PlotTick> YTicks { get; set; } = Array.Empty<PlotTick>();
  /// <summary>x of the vertical marker; NaN = none.</summary>
  public double MarkerX { get; set; } = double.NaN;
  public string MarkerLabel { get; set; } = "";
}

/// <summary>
/// The Model tab's dust curves: Q_d(r_h) with the engine's law (<c>particles.rs</c>
/// <c>dust_production_rate_kgs</c>) and the grain size law n(s) ∝ s^−q between s₀/F and s₀·F.
/// </summary>
public static class DustCurves
{
  public const int DustRateSamples = 240;
  public const int SizeSamples = 120;
  /// <summary>Q_d plot x range starts here (AU).</summary>
  public const double DustRateXMinAu = 0.5;

  private static bool IsFinite(double v) => !double.IsNaN(v) && !double.IsInfinity(v);

  // ── Q_d(r_h) ──────────────────────────────────────────────────────────────────

  /// <summary>
  /// Dust production rate Q_d (kg/s) at heliocentric distance <paramref name="rAu"/>, exactly as
  /// the engine: 0 at or beyond the cutoff, else min(max, Q_d₀ · r^−k).
  /// </summary>
  public static double DustRateKgs(double rAu, double rate1AuKgs, double power, double cutoffAu, double maxKgs)
  {
    if (!(rAu > 0) || rAu >= cutoffAu)
      return 0.0;
    double q = rate1AuKgs * Math.Pow(rAu, -power);
    return Math.Min(q, maxKgs);
  }

  /// <summary>
  /// Q_d(r_h) samples over [0.5 AU, cutoff + 1 AU]: uniform in r, plus the clamp kink and the
  /// last point before the cutoff and the cutoff itself (0), so the drop is drawn vertical.
  /// </summary>
  public static IReadOnlyList<PlotPoint> SampleDustRate(double rate1AuKgs, double power, double cutoffAu, double maxKgs)
  {
    double x0 = DustRateXMinAu;
    double x1 = DustRateXMaxAu(cutoffAu);
    var xs = new List<double>(DustRateSamples + 4);
    for (int i = 0; i <= DustRateSamples; i++)
      xs.Add(x0 + (x1 - x0) * i / DustRateSamples);
    // clamp kink: Q_d₀ r^−k = max ⇔ r = (Q_d₀ / max)^(1/k)
    if (power > 0 && maxKgs > 0 && rate1AuKgs > 0)
    {
      double rk = Math.Pow(rate1AuKgs / maxKgs, 1.0 / power);
      if (rk > x0 && rk < x1)
        xs.Add(rk);
    }
    if (cutoffAu > x0 && cutoffAu < x1)
    {
      xs.Add(cutoffAu * (1 - 1e-9));
      xs.Add(cutoffAu);
    }
    xs.Sort();
    var pts = new List<PlotPoint>(xs.Count);
    double prev = double.NaN;
    foreach (double x in xs)
    {
      if (x == prev) continue;
      prev = x;
      pts.Add(new PlotPoint(x, DustRateKgs(x, rate1AuKgs, power, cutoffAu, maxKgs)));
    }
    return pts;
  }

  public static double DustRateXMaxAu(double cutoffAu)
  {
    double x1 = (IsFinite(cutoffAu) ? cutoffAu : 15.0) + 1.0;
    return Math.Max(x1, DustRateXMinAu + 1.0);
  }

  /// <summary>The Q_d(r_h) plot: linear r_h axis, log Q_d axis, marker at the comet's r_h (NaN = none).</summary>
  public static CurvePlotModel DustRatePlot(double rate1AuKgs, double power, double cutoffAu, double maxKgs, double rhNowAu)
  {
    var pts = SampleDustRate(rate1AuKgs, power, cutoffAu, maxKgs);
    double xMin = DustRateXMinAu, xMax = DustRateXMaxAu(cutoffAu);
    var (yMin, yMax) = LogRange(pts);
    var model = new CurvePlotModel
    {
      Series = new[] { new PlotSeries("Q_d(r_h)", pts) },
      XMin = xMin,
      XMax = xMax,
      YMin = yMin,
      YMax = yMax,
      XLog = false,
      YLog = true,
      XLabel = "r_h (AU)",
      YLabel = "Q_d (kg/s)",
      XTicks = LinearTicks(xMin, xMax, 5),
      YTicks = LogTicks(yMin, yMax),
    };
    if (IsFinite(rhNowAu) && rhNowAu >= xMin && rhNowAu <= xMax)
    {
      model.MarkerX = rhNowAu;
      model.MarkerLabel = "now " + rhNowAu.ToString("0.00", CultureInfo.InvariantCulture) + " AU";
    }
    return model;
  }

  /// <summary>
  /// The one-line read-out under the Q_d plot: "Q_d now (r_h X AU): Y kg/s", of jet
  /// <paramref name="jetIndex"/> (0 = the defaults for new jets); <paramref name="rhNowAu"/> NaN =
  /// no committed comet.
  /// </summary>
  public static string DustRateNowText(double rhNowAu, double rate1AuKgs, double power, double cutoffAu, double maxKgs,
    int jetIndex)
  {
    if (!IsFinite(rhNowAu) || !(rhNowAu > 0))
      return "Q_d now: commit a comet to see its current distance from the Sun.";
    var ci = CultureInfo.InvariantCulture;
    double q = DustRateKgs(rhNowAu, rate1AuKgs, power, cutoffAu, maxKgs);
    string text = "Q_d now (r_h " + rhNowAu.ToString("0.00", ci) + " AU): " + FormatNumber(q) + " kg/s";
    text += jetIndex > 0 ? " (jet " + jetIndex.ToString(ci) + ")" : " (defaults for new jets)";
    return text;
  }

  /// <summary>
  /// The whole comet's activity: Σ Q_d over the jets (each with its own Q_d₀, the law's shape
  /// shared), at 1 AU and, with a committed comet, at <paramref name="rhNowAu"/>. The rates add up:
  /// each jet's coma brightness is linear in its production.
  /// </summary>
  public static string TotalActivityText(IReadOnlyList<double> jetRates1AuKgs,
    double power, double cutoffAu, double maxKgs, double rhNowAu)
  {
    if (jetRates1AuKgs.Count == 0)
      return "Σ activity: no jets.";
    var ci = CultureInfo.InvariantCulture;
    double q1 = 0, qNow = 0;
    bool now = IsFinite(rhNowAu) && rhNowAu > 0;
    foreach (double q0 in jetRates1AuKgs)
    {
      q1 += DustRateKgs(1.0, q0, power, cutoffAu, maxKgs);
      if (now)
        qNow += DustRateKgs(rhNowAu, q0, power, cutoffAu, maxKgs);
    }
    int n = jetRates1AuKgs.Count;
    string text = "Σ " + n.ToString(ci) + (n == 1 ? " jet" : " jets") + " at 1 AU: Σ Q_d " + FormatNumber(q1) + " kg/s";
    if (now)
      text += "; now (r_h " + rhNowAu.ToString("0.00", ci) + " AU): Σ Q_d " + FormatNumber(qNow) + " kg/s";
    return text;
  }
  // ── grain size law ───────────────────────────────────────────────────────────

  /// <summary>Smallest F the plots use: F = 1 is a single size (a delta), not a curve.</summary>
  public const double MinPlotSizeRange = 1.01;

  /// <summary>Grain radius bounds s₀/F, s₀·F in µm (s₀ = diameter / 2).</summary>
  public static (double Min, double Max) RadiusBoundsUm(double diameterUm, double sizeRange)
  {
    double s0 = diameterUm * 0.5;
    double f = Math.Max(sizeRange, MinPlotSizeRange);
    return (s0 / f, s0 * f);
  }

  /// <summary>
  /// Normalised power law density ∝ s^k on [a, b]: ∫ₐᵇ c s^k ds = 1.
  /// </summary>
  public static double PowerLawNorm(double k, double a, double b)
  {
    double e = k + 1.0;
    if (Math.Abs(e) < 1e-9)
      return 1.0 / Math.Log(b / a);
    return e / (Math.Pow(b, e) - Math.Pow(a, e));
  }

  /// <summary>Number density n(s) ∝ s^−q on [s₀/F, s₀·F], unit area, per µm; 0 outside.</summary>
  public static double SizePdf(double sUm, double diameterUm, double sizeRange, double q)
  {
    var (a, b) = RadiusBoundsUm(diameterUm, sizeRange);
    if (sUm < a || sUm > b) return 0.0;
    return PowerLawNorm(-q, a, b) * Math.Pow(sUm, -q);
  }

  /// <summary>Cross-section-weighted density s²·n(s) ∝ s^(2−q), unit area, per µm; 0 outside.</summary>
  public static double CrossSectionPdf(double sUm, double diameterUm, double sizeRange, double q)
  {
    var (a, b) = RadiusBoundsUm(diameterUm, sizeRange);
    if (sUm < a || sUm > b) return 0.0;
    return PowerLawNorm(2.0 - q, a, b) * Math.Pow(sUm, 2.0 - q);
  }

  private static IReadOnlyList<PlotPoint> SampleLog(double a, double b, Func<double, double> f)
  {
    var pts = new List<PlotPoint>(SizeSamples + 1);
    double la = Math.Log(a), lb = Math.Log(b);
    for (int i = 0; i <= SizeSamples; i++)
    {
      // the ends exactly (exp∘log may step just outside)
      double s = i == 0 ? a : i == SizeSamples ? b : Math.Exp(la + (lb - la) * i / SizeSamples);
      pts.Add(new PlotPoint(s, f(s)));
    }
    return pts;
  }

  /// <summary>
  /// The size distribution plot: by number n(s) and by cross-section s²n(s) (what the brightness
  /// follows), both normalised to unit area, log–log over [s₀/F, s₀·F].
  /// </summary>
  public static CurvePlotModel SizeDistributionPlot(double diameterUm, double sizeRange, double q)
  {
    var (a, b) = RadiusBoundsUm(diameterUm, sizeRange);
    var byNumber = SampleLog(a, b, s => SizePdf(s, diameterUm, sizeRange, q));
    var byArea = SampleLog(a, b, s => CrossSectionPdf(s, diameterUm, sizeRange, q));
    var all = new List<PlotPoint>(byNumber.Count + byArea.Count);
    all.AddRange(byNumber);
    all.AddRange(byArea);
    var (yMin, yMax) = LogRange(all);
    return new CurvePlotModel
    {
      Series = new[]
      {
        new PlotSeries("by number", byNumber, 0),
        new PlotSeries("by cross-section (brightness)", byArea, 1),
      },
      XMin = a,
      XMax = b,
      YMin = yMin,
      YMax = yMax,
      XLog = true,
      YLog = true,
      XLabel = "grain radius s (µm)",
      YLabel = "probability density (1/µm)",
      XTicks = LogTicks(a, b),
      YTicks = LogTicks(yMin, yMax),
    };
  }

  // ── axes ───────────────────────────────────────────────────────────────────────

  /// <summary>Decade-rounded range of the positive finite y values ((1, 10) when there are none).</summary>
  public static (double Min, double Max) LogRange(IEnumerable<PlotPoint> points)
  {
    double lo = double.PositiveInfinity, hi = double.NegativeInfinity;
    foreach (var p in points)
    {
      if (!(p.Y > 0) || !IsFinite(p.Y)) continue;
      lo = Math.Min(lo, p.Y);
      hi = Math.Max(hi, p.Y);
    }
    if (!IsFinite(lo) || !IsFinite(hi))
      return (1.0, 10.0);
    double dLo = Math.Floor(Math.Log10(lo));
    double dHi = Math.Ceiling(Math.Log10(hi));
    if (dHi <= dLo) dHi = dLo + 1;
    return (Math.Pow(10, dLo), Math.Pow(10, dHi));
  }

  /// <summary>
  /// Decade ticks inside [min, max] (every n-th decade when there are more than six); the two
  /// ends when no decade falls inside.
  /// </summary>
  public static IReadOnlyList<PlotTick> LogTicks(double min, double max)
  {
    var ticks = new List<PlotTick>();
    if (!(min > 0) || !(max > min)) return ticks;
    int d0 = (int)Math.Ceiling(Math.Log10(min) - 1e-9);
    int d1 = (int)Math.Floor(Math.Log10(max) + 1e-9);
    int count = d1 - d0 + 1;
    int step = Math.Max(1, (int)Math.Ceiling(count / 6.0));
    for (int d = d0; d <= d1; d += step)
    {
      double v = Math.Pow(10, d);
      ticks.Add(new PlotTick(v, FormatNumber(v)));
    }
    if (ticks.Count < 2)
    {
      ticks.Clear();
      ticks.Add(new PlotTick(min, FormatNumber(min)));
      ticks.Add(new PlotTick(max, FormatNumber(max)));
    }
    return ticks;
  }

  /// <summary>"Nice" linear ticks (1, 2, 5 × 10ⁿ steps) inside [min, max], about <paramref name="target"/> of them.</summary>
  public static IReadOnlyList<PlotTick> LinearTicks(double min, double max, int target)
  {
    var ticks = new List<PlotTick>();
    if (!(max > min) || target < 1) return ticks;
    double raw = (max - min) / target;
    double mag = Math.Pow(10, Math.Floor(Math.Log10(raw)));
    double norm = raw / mag;
    double step = (norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10) * mag;
    double first = Math.Ceiling(min / step - 1e-9) * step;
    for (double v = first; v <= max + step * 1e-9; v += step)
    {
      double r = Math.Round(v / step) * step;
      ticks.Add(new PlotTick(r, FormatNumber(r)));
    }
    return ticks;
  }

  /// <summary>Compact invariant number: plain between 0.01 and 10⁴, else "1.2e5".</summary>
  public static string FormatNumber(double v)
  {
    var ci = CultureInfo.InvariantCulture;
    if (v == 0) return "0";
    if (!IsFinite(v)) return v.ToString(ci);
    double av = Math.Abs(v);
    if (av >= 0.01 && av < 1e4)
      return v.ToString(av >= 100 ? "0" : av >= 1 ? "0.##" : "0.###", ci);
    int e = (int)Math.Floor(Math.Log10(av));
    double m = v / Math.Pow(10, e);
    // log10 rounding can land one decade off
    if (Math.Abs(m) >= 10 - 1e-9) { m /= 10; e++; }
    else if (Math.Abs(m) < 1 - 1e-9) { m *= 10; e--; }
    string ms = Math.Abs(Math.Abs(m) - 1) < 1e-9 ? (m < 0 ? "-1" : "1") : m.ToString("0.#", ci);
    return ms + "e" + e.ToString(ci);
  }
}
