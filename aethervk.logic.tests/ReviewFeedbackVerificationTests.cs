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
/// items) that no other test asserted on the current tree. The view items are checked on the loaded
/// views in aethervk.app.tests (<c>ReviewFeedbackViewTests</c>).
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

  // ── mail ─────────────────────────────────────────────────────────────────────

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
  /// docx 8: numbers print with a dot on an Italian locale once <see cref="NumberCulture.Apply"/>
  /// (what Program.Main runs at startup) has run, and the date keeps the locale. Without it the same
  /// formatting path prints a comma, so the swap is what the user sees.
  /// </summary>
  [Fact]
  public void Review_docx_8_NumbersUseADot_UnderAnItalianLocale()
  {
    var previous = Thread.CurrentThread.CurrentCulture;
    var previousDefault = CultureInfo.DefaultThreadCurrentCulture;
    try
    {
      var italian = CultureInfo.GetCultureInfo("it-IT");
      Thread.CurrentThread.CurrentCulture = italian;
      Assert.Equal("0,5", ViewportOverlayViewModel.FormatNiceNumber(0.5)); // the complaint

      // what Program.Main runs at startup
      var culture = NumberCulture.Apply();
      Assert.Same(culture, Thread.CurrentThread.CurrentCulture);
      Assert.Same(culture, CultureInfo.DefaultThreadCurrentCulture);
      Assert.Equal("it-IT", culture.Name);
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
      CultureInfo.DefaultThreadCurrentCulture = previousDefault;
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
}

