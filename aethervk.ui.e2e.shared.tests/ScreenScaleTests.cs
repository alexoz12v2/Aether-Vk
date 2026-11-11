using AetherVk.UI.E2E.Shared;

namespace AetherVk.UI.E2E.Shared.Tests;

public sealed class ScreenScaleTests
{
  [Theory]
  [InlineData(2880, 1440, 2.0)]
  [InlineData(1920, 1920, 1.0)]
  [InlineData(3024, 1512, 2.0)]
  [InlineData(0, 1440, 1.0)]
  [InlineData(2880, 0, 1.0)]
  public void FromModes_IsThePixelPerPointRatio(long pixels, long points, double scale) =>
    Assert.Equal(scale, ScreenScale.FromModes(pixels, points), 12);

  [Fact]
  public void ToPixels_OnRetina_DoublesWholePoints() =>
    Assert.Equal(new E2ERect(20, 40, 200, 100), ScreenScale.ToPixels(new PointRect(10, 20, 100, 50), 2));

  [Fact]
  public void ToPixels_CoversEveryTouchedPixel()
  {
    // 10.5 → 21, 20.25 → 40.5 (floor 40); right 111 → 222, bottom 70.5 → 141
    Assert.Equal(new E2ERect(21, 40, 201, 101), ScreenScale.ToPixels(new PointRect(10.5, 20.25, 100.5, 50.25), 2));
    // a non-integer scale: 1.5 → 1, right 6 → 6
    Assert.Equal(new E2ERect(1, 1, 5, 5), ScreenScale.ToPixels(new PointRect(1, 1, 3, 3), 1.5));
  }

  [Fact]
  public void ToPixels_AtScaleOne_IsTheSameRectangle() =>
    Assert.Equal(new E2ERect(7, 9, 300, 200), ScreenScale.ToPixels(new PointRect(7, 9, 300, 200), 1));

  [Fact]
  public void ToPixels_IgnoresFloatingPointNoise() =>
    // 0.1 * 3 is 0.30000000000000004: still 3 px at scale 10
    Assert.Equal(new E2ERect(1, 0, 2, 3), ScreenScale.ToPixels(new PointRect(0.1, 0, 0.1 * 2, 0.1 * 3), 10));

  [Fact]
  public void ToPoints_InvertsToPixels()
  {
    var px = new E2ERect(20, 40, 200, 100);
    var pt = ScreenScale.ToPoints(px, 2);
    Assert.Equal(new PointRect(10, 20, 100, 50), pt);
    Assert.Equal(px, ScreenScale.ToPixels(pt, 2));
  }

  [Fact]
  public void PointAndPixel_Conversions()
  {
    Assert.Equal((21, 41), ScreenScale.PointToPixel(10.5, 20.4, 2));
    Assert.Equal((10.5, 20.0), ScreenScale.PixelToPoint(21, 40, 2));
  }

  [Fact]
  public void Resample_SameSize_IsACopy()
  {
    var src = new[] { 1.0, 2, 3, 4 };
    var dst = ScreenScale.Resample(src, 2, 2, 2, 2);
    Assert.Equal(src, dst);
    Assert.NotSame(src, dst);
  }

  [Fact]
  public void Resample_HalfSize_PicksTheNearestSourcePixel()
  {
    // 4 × 4 → 2 × 2: destination centres 0.5 and 1.5 map to source 1 and 3
    var src = Enumerable.Range(0, 16).Select(i => (double)i).ToArray();
    Assert.Equal(new[] { 5.0, 7, 13, 15 }, ScreenScale.Resample(src, 4, 4, 2, 2));
  }

  [Fact]
  public void Resample_DoubleSize_RepeatsPixels()
  {
    var src = new[] { 1.0, 2, 3, 4 };
    Assert.Equal(new[] { 1.0, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4 }, ScreenScale.Resample(src, 2, 2, 4, 4));
  }

  [Fact]
  public void Resample_RejectsAPlaneOfTheWrongSize() =>
    Assert.Throws<ArgumentException>(() => ScreenScale.Resample(new double[3], 2, 2, 1, 1));
}
