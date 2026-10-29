using System;
using System.Globalization;
using System.Threading;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The viewport scale bar label must never read "0 &lt;unit&gt;", however far the view is zoomed.
/// </summary>
public class ViewportOverlayMeasurementTests
{
  [Fact]
  public void FormatNiceNumber_NeverPrintsZero()
  {
    foreach (double mantissa in new[] { 1.0, 2.0, 5.0 })
    {
      for (int exponent = -12; exponent <= 6; exponent++)
      {
        double nice = ViewportOverlayViewModel.GetNiceNumber(mantissa * Math.Pow(10, exponent));
        string text = ViewportOverlayViewModel.FormatNiceNumber(nice);
        double parsed = double.Parse(text, CultureInfo.CurrentCulture);
        Assert.True(parsed > 0, $"{mantissa}e{exponent} formatted as \"{text}\"");
        Assert.Equal(nice, parsed, nice * 1e-6);
      }
    }
  }

  [Theory]
  [InlineData(2e-6, "0.000002")]
  [InlineData(1e-6, "0.000001")]
  [InlineData(5e-3, "0.005")]
  [InlineData(0.5, "0.5")]
  [InlineData(20.0, "20")]
  public void FormatNiceNumber_UsesExactlyTheNeededDecimals(double value, string expected)
  {
    var previous = Thread.CurrentThread.CurrentCulture;
    Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
    try
    {
      Assert.Equal(expected, ViewportOverlayViewModel.FormatNiceNumber(value));
    }
    finally
    {
      Thread.CurrentThread.CurrentCulture = previous;
    }
  }

  private const double AuKm = 149_597_870.7;

  private static CameraProjectionState Ortho(double widthKm) =>
    new(false, 0f, 1.6f, 0f, 1f, (float)(-widthKm / 2 / AuKm), (float)(widthKm / 2 / AuKm), -1f, 1f, 1f);

  /// Reviewer report 2026-10-29: a 61 000 km field read in AU. Km now reach 10⁷ km.
  [Theory]
  [InlineData(61_000.0, "km")]
  [InlineData(1.5e6, "km")]
  [InlineData(5e7, "km")]       // bar ≈ 3.5e6 km
  [InlineData(1.5e8, "AU")]     // bar ≈ 1.05e7 km: past the limit
  [InlineData(1.5e9, "AU")]
  [InlineData(2.0, "m")]
  public void Ortho_ShowsKmUpToTenMillionKm(double fieldKm, string unit)
  {
    var m = ViewportOverlayViewModel.Measure(Ortho(fieldKm), 1000.0, null);
    Assert.NotNull(m);
    double barKm = fieldKm * 0.07;
    string expected = barKm >= ViewportOverlayViewModel.MaxKmOnScaleBar ? "AU" : unit;
    Assert.Equal(expected, m!.Value.Unit);
    // the bar's px width is the nice length at the field's scale
    double kmPerPx = fieldKm / 1000.0;
    double niceKm = m.Value.Unit switch { "AU" => m.Value.Nice * AuKm, "km" => m.Value.Nice, "m" => m.Value.Nice / 1e3, _ => double.NaN };
    Assert.Equal(niceKm / kmPerPx, m.Value.WidthPx, m.Value.WidthPx * 1e-5);
  }

  [Fact]
  public void EarthObserver_GivesTheKmTheBarSpansAtTheComet()
  {
    // 30' wide field at Δ = 0.44 AU (67P on 2021-12-27): 1000 px, bar ≥ 70 px = 2.1′ → nice 5′
    var proj = new CameraProjectionState(true, (float)(30.0 / 60.0 * Math.PI / 180.0), 1f, 1e-6f, 10f, 0, 0, 0, 0, 1f);
    var m = ViewportOverlayViewModel.Measure(proj, 1000.0, 0.44);
    Assert.NotNull(m);
    Assert.Equal("arcmin", m!.Value.Unit);
    Assert.Equal(5.0, m.Value.Nice);
    Assert.Equal(1000.0 * 5.0 / 30.0, m.Value.WidthPx, 1e-3);
    // 5′ at 0.44 AU = 0.44 × 1.496e8 × 1.454e-3 ≈ 95 700 km
    Assert.StartsWith(" ≈ ", m.Value.Suffix);
    Assert.EndsWith(" km at the comet", m.Value.Suffix);
    double km = 2 * 0.44 * AuKm * Math.Tan(5.0 / 60.0 * Math.PI / 180.0 / 2);
    Assert.InRange(km, 95_500, 96_000);
    Assert.Equal(ViewportOverlayViewModel.FormatKm(km), m.Value.Suffix[3..^" at the comet".Length]);

    // no distance (not the Earth observer): angle only
    Assert.Equal(string.Empty, ViewportOverlayViewModel.Measure(proj, 1000.0, null)!.Value.Suffix);
  }

  [Fact]
  public void FormatKm_KeepsThreeSignificantDigits()
  {
    var previous = Thread.CurrentThread.CurrentCulture;
    Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
    try
    {
      Assert.Equal("38,300 km", ViewportOverlayViewModel.FormatKm(38_291.4));
      Assert.Equal("123 km", ViewportOverlayViewModel.FormatKm(123.4));
      Assert.Equal("42.2 km", ViewportOverlayViewModel.FormatKm(42.24));
      Assert.Equal("4.2 km", ViewportOverlayViewModel.FormatKm(4.2));
    }
    finally
    {
      Thread.CurrentThread.CurrentCulture = previous;
    }
  }
}
