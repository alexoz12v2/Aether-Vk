using AetherVk.UI.E2E.Shared;

namespace AetherVk.UI.E2E.Shared.Tests;

public sealed class FrameMetricsCoreTests
{
  /// <summary>A capture of a synthetic screen: <paramref name="f"/>(x, y) per pixel of the rectangle.</summary>
  private static Func<E2ERect, double[]> Screen(Func<int, int, double> f) => r =>
  {
    var lum = new double[r.Width * r.Height];
    for (int j = 0; j < r.Height; j++)
      for (int i = 0; i < r.Width; i++)
        lum[j * r.Width + i] = f(r.X + i, r.Y + j);
    return lum;
  };

  [Fact]
  public void Central_IsTheCentredNinetyPercent_InIntegerPixels()
  {
    // the Windows probe's arithmetic: (int)(801 * 0.9) = 720, offset (801 - 720) / 2 = 40
    Assert.Equal(new E2ERect(140, 80, 720, 540), FrameMetricsCore.Central(new E2ERect(100, 50, 801, 601)));
    Assert.Equal(new E2ERect(5, 5, 90, 90), FrameMetricsCore.Central(new E2ERect(0, 0, 100, 100)));
  }

  [Fact]
  public void Grid_PicksTheIntegerCellCentres()
  {
    const int w = 64, h = 32, n = 4;
    var full = Enumerable.Range(0, w * h).Select(k => (double)k).ToArray();
    var grid = FrameMetricsCore.Grid(full, w, h, n);
    Assert.Equal(n * n, grid.Length);
    for (int j = 0; j < n; j++)
      for (int i = 0; i < n; i++)
        Assert.Equal((2 * j + 1) * h / (2 * n) * w + (2 * i + 1) * w / (2 * n), grid[j * n + i]);
    // row 4, column 8
    Assert.Equal(4 * 64 + 8, grid[0]);
  }

  [Fact]
  public void Grid_RejectsAShortCapture() =>
    Assert.Throws<ArgumentException>(() => FrameMetricsCore.Grid(new double[10], 4, 4, 2));

  [Fact]
  public void Sample_AsksTheDriverForTheCentralRectangleOnTheGrid()
  {
    E2ERect asked = default;
    int askedN = 0;
    var grid = FrameMetricsCore.Sample((r, n) => { asked = r; askedN = n; return new double[n * n]; }, new E2ERect(0, 0, 100, 200));
    Assert.Equal(new E2ERect(5, 10, 90, 180), asked);
    Assert.Equal(FrameMetricsCore.GridSize, askedN);
    Assert.Equal(32 * 32, grid.Length);
  }

  [Fact]
  public void Changed_IsTheShareOfCellsMovedByMoreThanTheThreshold()
  {
    var a = new double[4];
    var b = new[] { 0.005, 0.0101, 0.5, 0.01 };
    Assert.Equal(0.5, FrameMetricsCore.Changed(a, b), 12);
    Assert.Equal(0.0, FrameMetricsCore.Changed(b, b));
  }

  [Fact]
  public void Changed_RejectsGridsOfDifferentSizes() =>
    Assert.Throws<ArgumentException>(() => FrameMetricsCore.Changed(new double[4], new double[5]));

  [Fact]
  public void RingMean_AveragesOnlyTheRing()
  {
    const int inner = 3, outer = 6, size = 2 * outer + 1;
    var box = new double[size * size];
    for (int j = 0; j < size; j++)
      for (int i = 0; i < size; i++)
      {
        int dx = i - outer, dy = j - outer, r2 = dx * dx + dy * dy;
        box[j * size + i] = r2 < inner * inner ? 9 : r2 > outer * outer ? 5 : 1;
      }
    Assert.Equal(1.0, FrameMetricsCore.RingMean(box, inner, outer), 12);
  }

  [Fact]
  public void RingAndSky_RingAroundTheCentre_SkyIsTheDarkestCorner()
  {
    var vp = new E2ERect(0, 0, 1000, 800);
    (double X, double Y) comet = (500, 400);
    var capture = Screen((x, y) =>
    {
      double dx = x - comet.X, dy = y - comet.Y;
      if (dx * dx + dy * dy <= 60 * 60)
        return 0.8;
      return x < 100 && y < 100 ? 0.3 : 0.1; // an overlay panel in the top-left corner
    });
    var (ring, sky) = FrameMetricsCore.RingAndSky(capture, vp, comet);
    Assert.Equal(0.8, ring, 12);
    Assert.Equal(0.1, sky, 12);
  }

  [Fact]
  public void SkyCorners_AreInsetIntoTheFourCorners()
  {
    var c = FrameMetricsCore.SkyCorners(new E2ERect(10, 20, 200, 100));
    Assert.Equal(new E2ERect(18, 28, 40, 40), c[0]);
    Assert.Equal(new E2ERect(162, 28, 40, 40), c[1]);
    Assert.Equal(new E2ERect(18, 72, 40, 40), c[2]);
    Assert.Equal(new E2ERect(162, 72, 40, 40), c[3]);
  }

  private static double[] Disc(int w, int h, double cx, double cy, double r)
  {
    var lum = new double[w * h];
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
        lum[y * w + x] = (x + 0.5 - cx) * (x + 0.5 - cx) + (y + 0.5 - cy) * (y + 0.5 - cy) <= r * r ? 0.7 : 0.01;
    return lum;
  }

  [Fact]
  public void BrightCentroid_OfACentredDisc_IsTheCentre()
  {
    var (x, y, count) = FrameMetricsCore.BrightCentroid(Disc(200, 100, 100, 50, 20), 200, 100, 0.05);
    Assert.InRange(x, 0.499, 0.501);
    Assert.InRange(y, 0.499, 0.501);
    Assert.True(count > 1000);
    Assert.True(FrameMetricsCore.CentreOffset(x, y) <= 0.10);
  }

  [Fact]
  public void BrightCentroid_OfAnOffsetDisc_IsOffCentre()
  {
    var (x, y, _) = FrameMetricsCore.BrightCentroid(Disc(200, 200, 150, 100, 10), 200, 200, 0.05);
    Assert.InRange(x, 0.74, 0.76);
    Assert.InRange(y, 0.49, 0.51);
    Assert.True(FrameMetricsCore.CentreOffset(x, y) > 0.10);
  }

  [Fact]
  public void BrightCentroid_OfABlackImage_IsNaN()
  {
    var (x, y, count) = FrameMetricsCore.BrightCentroid(new double[100], 10, 10, 0.05);
    Assert.True(double.IsNaN(x) && double.IsNaN(y));
    Assert.Equal(0, count);
  }

  [Fact]
  public void LitShare_And_Mean()
  {
    var v = new[] { 0.0, 0.06, 0.5, 0.04 };
    Assert.Equal(0.5, FrameMetricsCore.LitShare(v, 0.05), 12);
    Assert.Equal(0.15, FrameMetricsCore.Mean(v), 12);
    Assert.Equal(0.0, FrameMetricsCore.LitShare([], 0.05));
  }

  [Fact]
  public void Luma_IsRec709()
  {
    Assert.Equal(1.0, FrameMetricsCore.Luma(255, 255, 255), 12);
    Assert.Equal(0.0, FrameMetricsCore.Luma(0, 0, 0), 12);
    Assert.Equal(0.7152, FrameMetricsCore.Luma(0, 255, 0), 12);
    Assert.Equal(0.2126, FrameMetricsCore.Luma(255, 0, 0), 12);
  }
}
