using System;
using AetherVk.Logic.Models;
using AetherVk.Logic.Utils;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// <see cref="NucleusDetector"/> on synthetic comets: a Gaussian nucleus at a sub-pixel position
/// (image px, pixel centres at whole numbers) on a background with noise, a brighter star outside
/// the central third, blank pixels, and the same image as 8-bit texels.
/// </summary>
public class NucleusDetectorTests
{
  private const int W = 300, H = 240;

  /// A Gaussian (σ px, peak amplitude) at (x0, y0) on a background sloping by
  /// <paramref name="slope"/> per column with deterministic ±noise (an LCG), sampled at the pixel
  /// centres.
  private static float[] Comet(
    double x0, double y0, double sigma = 2.2, double amplitude = 1000, double noise = 5, double slope = 0.05)
  {
    var v = new float[W * H];
    uint seed = 12345;
    for (int j = 0; j < H; j++)
    for (int i = 0; i < W; i++)
    {
      seed = seed * 1664525u + 1013904223u;
      double n = ((seed >> 8) / (double)(1 << 24) - 0.5) * 2 * noise;
      double dx = i - x0, dy = j - y0;
      v[j * W + i] = (float)(100 + slope * i + amplitude * Math.Exp(-(dx * dx + dy * dy) / (2 * sigma * sigma)) + n);
    }
    return v;
  }

  private static void Within(double want, double got, double tol) =>
    Assert.True(Math.Abs(want - got) <= tol, $"expected {want} ± {tol}, got {got}");

  [Theory]
  [InlineData(151.37, 118.71)]
  [InlineData(140.5, 110.5)]
  [InlineData(160.02, 125.93)]
  public void Gaussian_IsFoundToATenthOfAPixel(double x0, double y0)
  {
    var p = NucleusDetector.Detect(new IntensityImage(W, H, Comet(x0, y0)))!.Value;
    Within(x0, p.X, 0.1);
    Within(y0, p.Y, 0.1);
  }

  /// A brighter star outside the central third does not win the peak search.
  [Fact]
  public void BrighterStarOutsideTheCentralThird_IsIgnored()
  {
    var v = Comet(151.37, 118.71);
    for (int j = 20; j < 30; j++)
    for (int i = 20; i < 30; i++)
      v[j * W + i] = 50_000;
    var p = NucleusDetector.Detect(new IntensityImage(W, H, v))!.Value;
    Within(151.37, p.X, 0.1);
    Within(118.71, p.Y, 0.1);
  }

  /// Blank pixels (NaN, a stack's uncovered border) are left out of every statistic.
  [Fact]
  public void BlankPixels_AreSkipped()
  {
    var v = Comet(151.37, 118.71);
    for (int j = 0; j < H; j++)
    for (int i = 0; i < 90; i++)
      v[j * W + i] = float.NaN;
    v[118 * W + 140] = float.NaN; // inside the centroid disc, off the core
    var p = NucleusDetector.Detect(new IntensityImage(W, H, v))!.Value;
    Within(151.37, p.X, 0.1);
    Within(118.71, p.Y, 0.1);
  }

  /// The same comet as grey RGBA8 texels (a JPEG / PNG): the 8-bit quantisation keeps it within a
  /// tenth of a pixel; a transparent texel is blank.
  [Fact]
  public void Rgba8Texels_GiveTheSameNucleus()
  {
    var v = Comet(151.37, 118.71, amplitude: 180, noise: 2, slope: 0.01);
    var rgba = new byte[W * H * 4];
    for (int k = 0; k < v.Length; k++)
    {
      byte g = (byte)Math.Max(0, Math.Min(255, Math.Round(v[k] - 60)));
      rgba[4 * k] = rgba[4 * k + 1] = rgba[4 * k + 2] = g;
      rgba[4 * k + 3] = 255;
    }
    rgba[3] = 0;
    var image = NucleusDetector.FromRgba8(W, H, rgba);
    Assert.True(float.IsNaN(image.Values[0]));
    var p = NucleusDetector.Detect(image)!.Value;
    Within(151.37, p.X, 0.1);
    Within(118.71, p.Y, 0.1);
  }

  /// A flat image has nothing above its background; an image too small for the box has no search.
  [Fact]
  public void NoSignalOrTooSmall_IsNull()
  {
    var flat = new float[W * H];
    Array.Fill(flat, 7f);
    Assert.Null(NucleusDetector.Detect(new IntensityImage(W, H, flat)));
    Assert.Null(NucleusDetector.Detect(new IntensityImage(6, 6, new float[36])));
  }
}
