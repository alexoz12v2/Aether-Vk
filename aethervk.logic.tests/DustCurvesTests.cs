using System;
using System.Linq;
using AetherVk.Logic.Utils;
using Xunit;

namespace AetherVk.Logic.Tests;

/// The Model tab's Q_d(r_h) and grain-size plots: the engine's production law and the normalised size law.
public class DustCurvesTests
{
  private static double LogSlope(Func<double, double> f, double x1, double x2) =>
    (Math.Log(f(x2)) - Math.Log(f(x1))) / (Math.Log(x2) - Math.Log(x1));

  [Fact]
  public void DustRate_At1Au_IsQ0()
  {
    Assert.Equal(2.0, DustCurves.DustRateKgs(1.0, 2, 2, 15, 2000), 9);
    Assert.Equal(50.0, DustCurves.DustRateKgs(1.0, 50, 3.1, 6, 1e6), 6);
  }

  [Fact]
  public void DustRate_IsClampedAtMax_NearTheSun()
  {
    // 2 · 0.01^-2 = 2e4 > 2000
    Assert.Equal(2000.0, DustCurves.DustRateKgs(0.01, 2, 2, 15, 2000), 9);
    var pts = DustCurves.SampleDustRate(1e3, 2, 15, 2e3);
    Assert.All(pts, p => Assert.True(p.Y <= 2e3 * (1 + 1e-12)));
    // the clamp kink r = (1e3/2e3)^(1/2) is sampled exactly
    Assert.Contains(pts, p => Math.Abs(p.X - Math.Sqrt(0.5)) < 1e-12 && Math.Abs(p.Y - 2e3) < 1e-6);
  }

  [Fact]
  public void DustRate_IsZero_AtAndBeyondTheCutoff()
  {
    Assert.Equal(0.0, DustCurves.DustRateKgs(15.0, 2, 2, 15, 2000));
    Assert.Equal(0.0, DustCurves.DustRateKgs(20.0, 2, 2, 15, 2000));
    Assert.True(DustCurves.DustRateKgs(14.999, 2, 2, 15, 2000) > 0);
    var pts = DustCurves.SampleDustRate(2, 2, 15, 2000);
    Assert.Equal(0.5, pts[0].X, 12);
    Assert.Equal(16.0, pts[^1].X, 12);
    Assert.All(pts.Where(p => p.X >= 15), p => Assert.Equal(0.0, p.Y));
    Assert.Contains(pts, p => p.X == 15.0);
    Assert.True(pts.Zip(pts.Skip(1), (a, b) => b.X > a.X).All(x => x));
  }

  [Theory]
  [InlineData(1.0)]
  [InlineData(2.0)]
  [InlineData(3.7)]
  public void DustRate_LogLogSlope_IsMinusPower(double k)
  {
    double slope = LogSlope(r => DustCurves.DustRateKgs(r, 2, k, 15, 1e9), 2, 4);
    Assert.Equal(-k, slope, 9);
  }

  [Fact]
  public void DustRatePlot_HasLogYDecades_AndTheMarkerInsideTheRange()
  {
    var m = DustCurves.DustRatePlot(2, 2, 15, 2000, 1.3);
    Assert.True(m.YLog);
    Assert.False(m.XLog);
    Assert.Equal(1.3, m.MarkerX, 12);
    Assert.True(m.YMin <= 2.0 / (15 * 15) && m.YMax >= 8);
    Assert.Equal("Q_d (kg/s)", m.YLabel);
    Assert.Equal("Q_d(r_h)", m.Series[0].Label);
    Assert.All(m.YTicks, t => Assert.InRange(t.Value, m.YMin, m.YMax));
    Assert.All(m.XTicks, t => Assert.InRange(t.Value, m.XMin, m.XMax));
    Assert.True(double.IsNaN(DustCurves.DustRatePlot(2, 2, 15, 2000, double.NaN).MarkerX));
    Assert.True(double.IsNaN(DustCurves.DustRatePlot(2, 2, 15, 2000, 40).MarkerX));
  }

  private static double Integrate(Func<double, double> f, double a, double b, int n = 20000)
  {
    // trapezoid in log s: ∫ f ds = ∫ f(s) s d(ln s)
    double la = Math.Log(a), lb = Math.Log(b), h = (lb - la) / n, sum = 0;
    for (int i = 0; i <= n; i++)
    {
      double s = Math.Exp(la + h * i);
      double w = i == 0 || i == n ? 0.5 : 1;
      sum += w * f(Math.Min(Math.Max(s, a), b)) * s;
    }
    return sum * h;
  }

  [Theory]
  [InlineData(100, 10, 3.5)]
  [InlineData(100, 10, 2.0)]
  [InlineData(20, 100, 5.0)]
  [InlineData(1, 3, 3.0)] // cross-section exponent −1: the log normalisation
  [InlineData(1, 3, 1.0)] // number exponent −1
  public void SizePdfs_IntegrateToOne(double d, double f, double q)
  {
    var (a, b) = DustCurves.RadiusBoundsUm(d, f);
    Assert.Equal(d / 2 / f, a, 12);
    Assert.Equal(d / 2 * f, b, 9);
    Assert.Equal(1.0, Integrate(s => DustCurves.SizePdf(s, d, f, q), a, b), 4);
    Assert.Equal(1.0, Integrate(s => DustCurves.CrossSectionPdf(s, d, f, q), a, b), 4);
  }

  [Theory]
  [InlineData(3.5)]
  [InlineData(2.5)]
  [InlineData(4.2)]
  public void SizePdf_SlopeIsMinusQ_CrossSectionSlopeIsTwoMinusQ(double q)
  {
    Assert.Equal(-q, LogSlope(s => DustCurves.SizePdf(s, 100, 10, q), 10, 100), 9);
    Assert.Equal(2 - q, LogSlope(s => DustCurves.CrossSectionPdf(s, 100, 10, q), 10, 100), 9);
  }

  [Fact]
  public void SizePlot_TwoLabelledLogLogCurves_OverTheRadiusRange()
  {
    var m = DustCurves.SizeDistributionPlot(100, 10, 3.5);
    Assert.True(m.XLog && m.YLog);
    Assert.Equal(5.0, m.XMin, 12);
    Assert.Equal(500.0, m.XMax, 9);
    Assert.Equal(new[] { "by number", "by cross-section (brightness)" }, m.Series.Select(s => s.Label));
    Assert.All(m.Series, s => Assert.All(s.Points, p => Assert.InRange(p.Y, m.YMin, m.YMax)));
    Assert.True(m.XTicks.Count >= 2);
    // F = 1 is a delta: plotted with the minimum spread, still finite
    var one = DustCurves.SizeDistributionPlot(100, 1, 3.5);
    Assert.All(one.Series, s => Assert.All(s.Points, p => Assert.True(p.Y > 0 && !double.IsInfinity(p.Y))));
  }

  /// <summary>
  /// Σ Q_d adds the jets (each its own Q_d₀, the law shared); at r_h each jet's rate follows the
  /// law, the cap applying per jet.
  /// </summary>
  [Fact]
  public void TotalActivityText_AddsTheJets()
  {
    Assert.Contains("no jets", DustCurves.TotalActivityText([], 2, 15, 2000, 1.0));
    var t = DustCurves.TotalActivityText([1.0, 3.0], 2, 15, 2000, 2.0);
    Assert.StartsWith("Σ 2 jets at 1 AU: Σ Q_d 4 kg/s", t);
    // at 2 AU, k = 2: every rate is a quarter
    Assert.EndsWith("now (r_h 2.00 AU): Σ Q_d 1 kg/s", t);
    // the cap applies per jet: 2 × min(2000, 800 · 0.5⁻²) = 4000
    Assert.Contains("Σ Q_d 4000 kg/s", DustCurves.TotalActivityText([800, 800], 2, 15, 2000, 0.5));
    Assert.Equal("Σ 2 jets at 1 AU: Σ Q_d 2 kg/s", DustCurves.TotalActivityText([1.0, 1.0], 2, 15, 2000, double.NaN));
  }

  [Fact]
  public void DustRateNowText_FollowsTheLaw_OrAsksForACommit()
  {
    Assert.Contains("commit", DustCurves.DustRateNowText(double.NaN, 10, 2, 15, 2000, 1));
    var t = DustCurves.DustRateNowText(2, 10, 2, 15, 2000, 1);
    Assert.StartsWith("Q_d now (r_h 2.00 AU): 2.5 kg/s", t);
    Assert.Contains("jet 1", t);
    Assert.Contains("defaults", DustCurves.DustRateNowText(2, 10, 2, 15, 2000, 0));
    Assert.Contains("0 kg/s", DustCurves.DustRateNowText(16, 10, 2, 15, 2000, 1));
  }

  [Theory]
  [InlineData(0.0, "0")]
  [InlineData(1.0, "1")]
  [InlineData(0.5, "0.5")]
  [InlineData(1000.0, "1000")]
  [InlineData(1e5, "1e5")]
  [InlineData(1e-3, "1e-3")]
  [InlineData(250000.0, "2.5e5")]
  public void FormatNumber_IsCompact(double v, string s) => Assert.Equal(s, DustCurves.FormatNumber(v));
}
