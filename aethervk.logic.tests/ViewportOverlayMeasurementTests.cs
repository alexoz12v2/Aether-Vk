using System;
using System.Globalization;
using System.Threading;
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
}
