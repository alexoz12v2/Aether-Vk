using System.Globalization;
using AetherVk.Logic.Utils;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>AETHERVK_UI_CULTURE: the interface language for documentation screenshots.</summary>
public class NumberCultureOverrideTests
{
  [Fact]
  public void AKnownCulture_BecomesTheUiCulture_AndApplyKeepsTheDotNumbers()
  {
    var (culture, ui) = (CultureInfo.CurrentCulture, CultureInfo.CurrentUICulture);
    var (defCulture, defUi) = (CultureInfo.DefaultThreadCurrentCulture, CultureInfo.DefaultThreadCurrentUICulture);
    try
    {
      CultureInfo.CurrentCulture = CultureInfo.CurrentUICulture = CultureInfo.GetCultureInfo("it-IT");
      Assert.True(NumberCulture.ApplyOverride(" en-US "));
      Assert.Equal("en-US", CultureInfo.CurrentUICulture.Name);
      NumberCulture.Apply();
      Assert.Equal("0.5", 0.5.ToString(CultureInfo.CurrentCulture));
    }
    finally
    {
      CultureInfo.CurrentCulture = culture;
      CultureInfo.CurrentUICulture = ui;
      CultureInfo.DefaultThreadCurrentCulture = defCulture;
      CultureInfo.DefaultThreadCurrentUICulture = defUi;
    }
  }

  [Theory]
  [InlineData(null)]
  [InlineData("")]
  [InlineData("not-a-culture-xyz")]
  public void EmptyOrUnknown_ChangesNothing(string? name)
  {
    var ui = CultureInfo.CurrentUICulture;
    Assert.False(NumberCulture.ApplyOverride(name));
    Assert.Same(ui, CultureInfo.CurrentUICulture);
  }
}
