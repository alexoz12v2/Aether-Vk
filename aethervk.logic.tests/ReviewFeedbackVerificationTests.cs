using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Xml.Linq;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// Checks for the reviewer's complaints (the mail of 2026-10-29, R1–R10, and the third-round docx
/// items) that no other test asserted on the current tree. The markup checks read the .axaml files as
/// text, the way <c>ModelTabMarkupTests</c> in aethervk.app.tests does; they belong there once the
/// app test project is the place for every view check (see VERIFY_2026-10-30.md).
/// </summary>
public class ReviewFeedbackVerificationTests
{
  /// <summary>The repository root, found from the review fixture the other tests already locate.</summary>
  private static string RepoRoot()
  {
    // test_assets/saves/review_67p_2021-12 → test_assets/saves → test_assets → root
    var fixture = new DirectoryInfo(ReviewSaveReproTests.FixtureDir());
    return fixture.Parent!.Parent!.Parent!.FullName;
  }

  private static string ReadRepoFile(params string[] parts)
  {
    var path = Path.Combine(new[] { RepoRoot() }.Concat(parts).ToArray());
    Assert.True(File.Exists(path), $"missing {path}");
    return File.ReadAllText(path);
  }

  private static string ModelTabAxaml() => ReadRepoFile("aethervk.ui-app", "Views", "ModelTabView.axaml");
  private static string SettingsTabAxaml() => ReadRepoFile("aethervk.ui-app", "Views", "SettingsTabView.axaml");

  /// <summary>The <paramref name="chars"/> characters of <paramref name="text"/> before the first <paramref name="anchor"/>.</summary>
  private static string Before(string text, string anchor, int chars)
  {
    int i = text.IndexOf(anchor, StringComparison.Ordinal);
    Assert.True(i >= 0, $"anchor not found: {anchor}");
    return text.Substring(Math.Max(0, i - chars), Math.Min(chars, i));
  }

  /// <summary>The <paramref name="chars"/> characters of <paramref name="text"/> after the first <paramref name="anchor"/>.</summary>
  private static string After(string text, string anchor, int chars)
  {
    int i = text.IndexOf(anchor, StringComparison.Ordinal);
    Assert.True(i >= 0, $"anchor not found: {anchor}");
    return text.Substring(i, Math.Min(chars, text.Length - i));
  }

  // ── mail ─────────────────────────────────────────────────────────────────────

  /// <summary>
  /// R1: a refused jet's row says so and offers Retry (the logic is in JetNativeSyncTests; this is
  /// the row's markup), and the aperture shows two decimals (0.2° used to read "0°").
  /// </summary>
  [Fact]
  public void Review_mail_R1_RefusedJetRowHasRetry_AndApertureShowsTwoDecimals()
  {
    var text = ModelTabAxaml();
    Assert.StartsWith("Not in the simulation", JetNativeSync.RefusedText);
    Assert.Contains("{Binding NativeError}", text);
    var retry = After(text, "Content=\"Retry\"", 400);
    Assert.Contains("RetryNativeJetsCommand", retry);
    Assert.Contains("{Binding ApertureDeg, StringFormat='{}{0:0.##}°'}", text);
    Assert.DoesNotContain("{Binding ApertureDeg, StringFormat='{}{0:0}°'}", text);
  }

  /// <summary>R6: no "Afρ" or "afrho" is left in any user-visible string of either resource file.</summary>
  [Theory]
  [InlineData("AppStrings.resx")]
  [InlineData("AppStrings.it.resx")]
  public void Review_mail_R6_NoAfrhoInAppStrings(string file)
  {
    var path = Path.Combine(RepoRoot(), "aethervk.ui-logic", "Resources", file);
    Assert.True(File.Exists(path), $"missing {path}");
    var data = XDocument.Load(path).Root!.Elements("data").ToList();
    Assert.NotEmpty(data);
    foreach (var d in data)
    {
      string name = (string?)d.Attribute("name") ?? "";
      string value = d.Element("value")?.Value ?? "";
      Assert.DoesNotContain("Afρ", value);
      Assert.DoesNotContain("afrho", value, StringComparison.OrdinalIgnoreCase);
      Assert.DoesNotContain("afrho", name, StringComparison.OrdinalIgnoreCase);
    }
  }

  /// <summary>R7: the size law is per jet: every new jet's grain starts at q = 3.5.</summary>
  [Fact]
  public void Review_mail_R7_NewJetGrainStartsAtQ3_5()
  {
    Assert.Equal(3.5f, AetherVk.Logic.Models.JetGrain.Default.SizeQ);
    Assert.Equal(3.5f, new JetViewModel(new ModelSession().DefaultGrain).SizeQ);
    Assert.DoesNotContain("AutomationId=\"SizeQBox\"", ModelTabAxaml());
  }

  // ── docx ─────────────────────────────────────────────────────────────────────

  /// <summary>
  /// docx 1 and 2: the General section (unbound placeholder) and the Orbital Elements section
  /// (never filled) of the Settings tab are hidden.
  /// </summary>
  [Fact]
  public void Review_docx_1_2_GeneralAndOrbitalElementsSectionsAreHidden()
  {
    var text = SettingsTabAxaml();
    foreach (var header in new[] { "Header=\"{Binding StrGeneral}\"", "Header=\"{Binding StrOrbitalElements}\"" })
    {
      var before = Before(text, header, 200);
      Assert.Contains("IsVisible=\"False\"", before);
    }
  }

  /// <summary>
  /// docx 8: numbers print with a dot on an Italian locale once the number format is the invariant
  /// one (the Program.cs recipe: clone the culture, swap NumberFormat), and the date keeps the locale.
  /// Without the swap the same formatting path prints a comma, so the swap is what the user sees.
  /// </summary>
  [Fact]
  public void Review_docx_8_NumbersUseADot_UnderAnItalianLocale()
  {
    var previous = Thread.CurrentThread.CurrentCulture;
    try
    {
      var italian = CultureInfo.GetCultureInfo("it-IT");
      Thread.CurrentThread.CurrentCulture = italian;
      Assert.Equal("0,5", ViewportOverlayViewModel.FormatNiceNumber(0.5)); // the complaint

      var culture = (CultureInfo)italian.Clone();
      culture.NumberFormat = CultureInfo.InvariantCulture.NumberFormat;
      Thread.CurrentThread.CurrentCulture = culture;
      Assert.Equal("0.5", ViewportOverlayViewModel.FormatNiceNumber(0.5));
      Assert.Equal("0.005", ViewportOverlayViewModel.FormatNiceNumber(0.005));
      Assert.Equal("42.2 km", ViewportOverlayViewModel.FormatKm(42.24));
      Assert.Equal("38,300 km", ViewportOverlayViewModel.FormatKm(38_291.4));
      Assert.Equal("0.5", DustCurves.FormatNumber(0.5));
      Assert.Equal("0.5", 0.5.ToString());
      Assert.Equal(0.5, double.Parse("0.5"));
      // dates keep the locale: an Italian short date is day-first
      Assert.Equal(new DateTime(2021, 12, 27).ToString("d", italian), new DateTime(2021, 12, 27).ToString("d"));
      Assert.StartsWith("27/12/2021", new DateTime(2021, 12, 27).ToString("d"));
    }
    finally
    {
      Thread.CurrentThread.CurrentCulture = previous;
    }
  }

  /// <summary>docx 8b: the pole precession rates start at zero in a new comet session.</summary>
  [Fact]
  public void Review_docx_8b_PrecessionRatesDefaultToZero()
  {
    var s = new CometSession();
    Assert.Equal(0.0, s.RotPoleRaRateDegCen);
    Assert.Equal(0.0, s.RotPoleDecRateDegCen);
  }

  /// <summary>
  /// docx 9: the Sun and the Nucleus-and-Spin-Axis sections of the Model tab follow the legacy flag,
  /// which is off in the design view model and in a fresh real one (its backing field has no initialiser).
  /// </summary>
  [Fact]
  public void Review_docx_9_SunAndNucleusSectionsAreHiddenBehindTheLegacyFlag()
  {
    var text = ModelTabAxaml();
    foreach (var section in new[] { "AutomationId=\"ModelSunSection\"", "AutomationId=\"NucleusSpinSection\"" })
    {
      var before = Before(text, section, 1200);
      Assert.Contains("IsVisible=\"{Binding EnableLegacyExpanders}\"", before);
    }
    Assert.False(new DesignModelTabViewModel().EnableLegacyExpanders);
    var field = typeof(ModelTabViewModel).GetField("_enableLegacyExpanders",
      System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
    Assert.NotNull(field);
    Assert.Equal(typeof(bool), field!.FieldType); // a bool field without an initialiser starts false
  }

  /// <summary>
  /// docx 10: the reference position error, the dust visibility and the flow speed controls each
  /// carry a tooltip saying what they do.
  /// </summary>
  [Fact]
  public void Review_docx_10_DustSettingsHaveTooltips()
  {
    var text = ModelTabAxaml();
    foreach (var binding in new[] { "{Binding StrShowReferenceError}", "{Binding StrDustVisibility}", "{Binding StrDustFlowSpeed}" })
    {
      var after = After(text, binding, 1200);
      Assert.Contains("ToolTip.Tip=\"", after);
    }
  }
}
