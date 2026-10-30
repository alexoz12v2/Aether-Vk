using OpenQA.Selenium.Appium;

namespace AetherVk.UI.E2E;

/// <summary>
/// What the viewport shows, measured off the screen: a coarse luminance grid compared frame to
/// frame (did the picture change, and how much), and the brightness around the comet against the
/// sky in the viewport's corners (is there dust at all).
/// </summary>
public static class FrameMetrics
{
  /// <summary>Grid cells per side of <see cref="Sample"/>.</summary>
  public const int GridSize = 32;

  /// <summary>A cell counts as changed when its luminance moved by more than this (0..1).</summary>
  public const double CellThreshold = 0.01;

  /// <summary>The viewport's central 90 %, on a 32 × 32 luminance grid (the nucleus stays centred).</summary>
  public static double[] Sample(AppiumWebElement vp)
  {
    int w = (int)(vp.Size.Width * 0.9), h = (int)(vp.Size.Height * 0.9);
    return Win32.SampleLuminance(vp.Location.X + (vp.Size.Width - w) / 2, vp.Location.Y + (vp.Size.Height - h) / 2, w, h, GridSize);
  }

  /// <summary>The share of cells whose luminance moved by more than <see cref="CellThreshold"/>.</summary>
  public static double Changed(double[] a, double[] b) =>
    a.Zip(b, (x, y) => Math.Abs(x - y) > CellThreshold ? 1.0 : 0.0).Average();

  /// <summary>
  /// The mean luminance of the ring <paramref name="innerPx"/>–<paramref name="outerPx"/> around
  /// the comet minus the sky: the darkest of four 40 × 40 px boxes inset 8 px into the viewport's
  /// corners (an overlay panel in one corner cannot raise it).
  /// </summary>
  public static (double Ring, double Sky) RingAndSky(AppiumWebElement vp, (double X, double Y) comet,
    int innerPx = 10, int outerPx = 60)
  {
    int cx = (int)Math.Round(comet.X), cy = (int)Math.Round(comet.Y);
    int size = 2 * outerPx + 1;
    var box = Win32.CaptureLuminance(cx - outerPx, cy - outerPx, size, size);
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

    const int Corner = 40, Inset = 8;
    int left = vp.Location.X + Inset, top = vp.Location.Y + Inset;
    int right = vp.Location.X + vp.Size.Width - Inset - Corner, bottom = vp.Location.Y + vp.Size.Height - Inset - Corner;
    double sky = new[] { (left, top), (right, top), (left, bottom), (right, bottom) }
      .Min(c => Win32.CaptureLuminance(c.Item1, c.Item2, Corner, Corner).Average());
    return (sum / n, sky);
  }
}
