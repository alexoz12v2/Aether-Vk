using AetherVk.Logic.Automation;

namespace AetherVk.UI.E2E.Shared.Tests;

/// <summary>The id contract the shared scenarios locate by (both drivers look up these strings).</summary>
public sealed class CompactAutomationIdsTests
{
  [Fact]
  public void Ids_AreUniqueAndNonEmpty()
  {
    Assert.All(CompactAutomationIds.All, id => Assert.False(string.IsNullOrWhiteSpace(id)));
    Assert.Equal(CompactAutomationIds.All.Length, CompactAutomationIds.All.Distinct(StringComparer.Ordinal).Count());
  }

  [Fact]
  public void Ids_NeverReuseTheMainViewportHost()
  {
    // the probes and the keyboard router tell the main view by "ViewportHost"
    Assert.DoesNotContain("ViewportHost", CompactAutomationIds.All);
    Assert.NotEqual("ViewportHost", CompactAutomationIds.NucleusMiniViewHost);
  }

  [Fact]
  public void Ids_AreUnderTheirPrefixes() =>
    Assert.All(CompactAutomationIds.All, id => Assert.True(
      id == CompactAutomationIds.CompactPanel || id.StartsWith("Compact.", StringComparison.Ordinal)
      || id.StartsWith("NucleusMiniView", StringComparison.Ordinal), $"\"{id}\" is outside the Compact.* / NucleusMiniView* names"));
}
