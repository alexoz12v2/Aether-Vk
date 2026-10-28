using Xunit;

namespace AetherVk.UI.E2E;

/// <summary>The ridge sampler on synthetic images (no app needed).</summary>
public class FanRidgesTests
{
  [Fact]
  public void PositionAngle_IsNorthUpEastLeft()
  {
    Assert.Equal(0, FanRidges.PositionAngle(0, -1), 6);   // up = north
    Assert.Equal(90, FanRidges.PositionAngle(-1, 0), 6);  // left = east
    Assert.Equal(180, FanRidges.PositionAngle(0, 1), 6);  // down = south
    Assert.Equal(270, FanRidges.PositionAngle(1, 0), 6);  // right = west
  }

  [Fact]
  public void TwoRadialRidges_AreFoundAtTheirPositionAngles()
  {
    const int n = 401;
    const double c = 200;
    var lum = new double[n * n];
    for (int y = 0; y < n; y++)
      for (int x = 0; x < n; x++)
      {
        double pa = FanRidges.PositionAngle(x - c, y - c);
        // a dim background, two 6°-wide ridges at PA 118° and 190°, the second brighter
        double v = 0.1;
        if (FanRidges.AngleBetween(pa, 118) < 3) v = 0.3;
        if (FanRidges.AngleBetween(pa, 190) < 3) v = 0.5;
        lum[y * n + x] = v;
      }
    var profile = FanRidges.Profile(lum, n, n, c, c, kmPerPx: 100, loKm: 5_000, hiKm: 15_000)!;
    var peaks = FanRidges.Peaks(profile);
    Assert.True(peaks.Count >= 2, FanRidges.Describe(profile));
    Assert.True(FanRidges.AngleBetween(peaks[0].Pa, 190) <= FanRidges.BinDeg, FanRidges.Describe(profile));
    Assert.True(FanRidges.AngleBetween(FanRidges.Nearest(peaks, 118)!.Value.Pa, 118) <= FanRidges.BinDeg);
    Assert.True(FanRidges.MaxIn(profile, 260, 290) < FanRidges.MinExcess);
  }

  [Fact]
  public void ClippedWedge_IsSaturated_AndItsEdgesAreFound()
  {
    const int n = 401;
    const double c = 200;
    var lum = new double[n * n];
    for (int y = 0; y < n; y++)
      for (int x = 0; x < n; x++)
      {
        double pa = FanRidges.PositionAngle(x - c, y - c);
        lum[y * n + x] = pa >= 104 && pa <= 197 ? 1.0 : 0.1; // a clipped wedge 104°–197°
      }
    var profile = FanRidges.Profile(lum, n, n, c, c, kmPerPx: 100, loKm: 5_000, hiKm: 15_000)!;
    Assert.True(FanRidges.IsSaturated(profile));
    var (lo, hi) = FanRidges.Edges(profile);
    Assert.True(FanRidges.AngleBetween(lo, 104) <= FanRidges.BinDeg, $"{lo}");
    Assert.True(FanRidges.AngleBetween(hi, 197) <= FanRidges.BinDeg, $"{hi}");
  }

  [Fact]
  public void Profile_IsNullOutsideTheImage()
  {
    var lum = new double[10 * 10];
    Assert.Null(FanRidges.Profile(lum, 10, 10, 5, 5, kmPerPx: 100, loKm: 50_000, hiKm: 60_000));
  }
}
