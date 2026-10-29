using AetherVk.Logic.SaveFile;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The pre-Q_d save conversion reproduces the engine's legacy Afρ → kg/s formula: the engine's own
/// goldens (its f32 port matches them to 2e-6).
/// </summary>
public class LegacyAfrhoTests
{
  [Theory]
  [InlineData(100, 100, 0.8, 2, 0.5, 0.04, 10, 1.5510886453350254)]
  [InlineData(100, 100, 0.533, 2, 0.5, 0.04, 10, 1.0334128099544606)]
  [InlineData(537, 250, 1.0, 3.1, 0.6, 0.05, 10, 33.05519924265463)]
  [InlineData(8.128, 20, 0.5, 1.5, 0.3, 0.03, 3, 0.013690648521032838)]
  [InlineData(100000, 100, 0.8, 2.5, 0.625, 0.04, 100, 3372.168918307603)]
  public void DustRate_MatchesTheEngineGoldens(double afrhoCm, double diametreUm, double densityGcm3,
    double vMeanMs, double vStdMs, double albedo, double sizeRange, double expectedKgs)
  {
    double q = LegacyAfrho.DustRateKgs(afrhoCm, diametreUm, densityGcm3, vMeanMs, vStdMs, albedo, sizeRange);
    Assert.True(System.Math.Abs(q - expectedKgs) <= 1e-12 * expectedKgs, $"{q} vs {expectedKgs}");
  }

  [Fact]
  public void DustRate_DegenerateInputs()
  {
    Assert.Equal(0.0, LegacyAfrho.DustRateKgs(100, 100, 0.8, -1, 0.5, 0.04, 10));
    Assert.Equal(0.0, LegacyAfrho.DustRateKgs(100, 100, 0.8, 2, 0.5, 0, 10));
    // v = 0: σ/v counts as 0, the rate is 0 through v
    Assert.Equal(0.0, LegacyAfrho.DustRateKgs(100, 100, 0.8, 0, 0.5, 0.04, 10));
    // F outside [1, 1000] falls back to 10
    Assert.Equal(LegacyAfrho.MeanSizeFactor(100, 10), LegacyAfrho.MeanSizeFactor(100, 5000));
    Assert.Equal(LegacyAfrho.MeanSizeFactor(100, 10), LegacyAfrho.MeanSizeFactor(100, double.NaN));
  }
}
