using System.Numerics;
using AetherVk.Logic.ViewModels;
using Xunit;

namespace AetherVk.Logic.Tests;

public class JetViewModelRestoreTests
{
  /// A saved jet comes back with every value and its seed (the dust and the scene's compatibility
  /// JSON depend on the seed).
  [Fact]
  public void RestoreConstructor_KeepsEveryValueAndTheSeed()
  {
    var j = new JetViewModel(12.5f, -40f, 22f, 2.5f, 0.6f, new Vector4(0.1f, 0.2f, 0.3f, 1f), false, 4_000_000_007u);
    Assert.Equal(12.5f, j.LatitudeDeg);
    Assert.Equal(-40f, j.LongitudeDeg);
    Assert.Equal(22f, j.ApertureDeg);
    Assert.Equal(2.5f, j.StartVelocityMeanMs);
    Assert.Equal(0.6f, j.StartVelocityStdMs);
    Assert.Equal(new Vector4(0.1f, 0.2f, 0.3f, 1f), j.StreamColor);
    Assert.False(j.IsPreviewVisible);
    Assert.Equal(4_000_000_007u, j.Seed);
  }
}
