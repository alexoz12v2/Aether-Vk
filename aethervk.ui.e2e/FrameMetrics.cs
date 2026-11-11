using AetherVk.UI.E2E.Shared;
using OpenQA.Selenium.Appium;

namespace AetherVk.UI.E2E;

/// <summary>
/// What the viewport shows, measured off the screen: a coarse luminance grid compared frame to
/// frame (did the picture change, and how much), and the brightness around the comet against the
/// sky in the viewport's corners (is there dust at all). The math is the shared
/// <see cref="FrameMetricsCore"/> (the Mac suite runs the same), over Win32 BitBlt captures.
/// </summary>
public static class FrameMetrics
{
  /// <summary>Grid cells per side of <see cref="Sample"/>.</summary>
  public const int GridSize = FrameMetricsCore.GridSize;

  /// <summary>A cell counts as changed when its luminance moved by more than this (0..1).</summary>
  public const double CellThreshold = FrameMetricsCore.CellThreshold;

  /// <summary>The viewport's central 90 %, on a 32 × 32 luminance grid (the nucleus stays centred).</summary>
  public static double[] Sample(AppiumWebElement vp) =>
    FrameMetricsCore.Sample((r, n) => Win32.SampleLuminance(r.X, r.Y, r.Width, r.Height, n), RectOf(vp));

  /// <summary>The share of cells whose luminance moved by more than <see cref="CellThreshold"/>.</summary>
  public static double Changed(double[] a, double[] b) => FrameMetricsCore.Changed(a, b);

  /// <summary>
  /// The mean luminance of the ring <paramref name="innerPx"/>–<paramref name="outerPx"/> around
  /// the comet minus the sky: the darkest of four 40 × 40 px boxes inset 8 px into the viewport's
  /// corners (an overlay panel in one corner cannot raise it).
  /// </summary>
  public static (double Ring, double Sky) RingAndSky(AppiumWebElement vp, (double X, double Y) comet,
    int innerPx = 10, int outerPx = 60) =>
    FrameMetricsCore.RingAndSky(r => Win32.CaptureLuminance(r.X, r.Y, r.Width, r.Height), RectOf(vp), comet, innerPx, outerPx);

  /// <summary>The element's rectangle in screen pixels (WinAppDriver's frame).</summary>
  public static E2ERect RectOf(AppiumWebElement e) => new(e.Location.X, e.Location.Y, e.Size.Width, e.Size.Height);
}
