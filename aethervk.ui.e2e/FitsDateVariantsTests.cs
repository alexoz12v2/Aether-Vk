using Xunit.Abstractions;

namespace AetherVk.UI.E2E;

/// <summary>
/// The date fixes on the committed date-variant FITS fixtures (test_assets/fits_date_variants):
/// a <c>DATE-OBS = '27/12/21'</c> (dd/mm/yy: read as 2021, with the panel's two-digit-year
/// selector offering the FITS standard's 1921) and a date-only <c>DATE-OBS</c> with the time in
/// <c>UT = '18 30 00'</c>. Each imports as a texture and becomes a billboard whose key facts show
/// the mid-exposure date, and the mode bar's Model works from that date instead of showing the
/// "no FITS" / "no readable date" hint. The review save is
/// opened for its scene; its own FITS billboard is deleted first so the fixture is the image the
/// observation chain reads. Artifacts go to TestResults/e2e/fits-date-variants/.
/// </summary>
public sealed class FitsDateVariantsTests(ITestOutputHelper output)
{
  private void Step(string text) => output.WriteLine($"{DateTime.Now:HH:mm:ss} {text}");

  /// <summary>The fixtures, the UTC mid-exposure their facts must show, and for a two-digit-year
  /// card the date the 19xx choice shows (null: a four-digit year, no selector).</summary>
  public static TheoryData<string, string, string?> Variants => new()
  {
    { "date_obs_dd_mm_yy.fits", "2021-12-27T00:00:00", "1921-12-27T00:00:00" },
    { "date_obs_date_only_ut.fits", "2021-12-27T18:30:00", null },
  };

  private static string FixturePath(string name) =>
    AetherVk.TestAssets.TestAssets.Require("fits_date_variants/" + name);

  /// <summary>The two hint texts Model shows without a usable image (AppStrings: ModeBar_NoFitsHint,
  /// ModeBar_FitsWithoutDateHint); the second's `{Date}` carries the raw card.</summary>
  private static readonly string[] NoImageHints = ["Import a FITS image", "no observation date"];

  /// <summary>The FITS bytes with the first pixel's last (least significant, big-endian) byte
  /// flipped in its lowest bit: the data unit starts at the 2880-byte block after the END card.</summary>
  private static byte[] NudgeFirstPixel(byte[] fits)
  {
    for (int card = 0; card + 80 <= fits.Length; card += 80)
    {
      if (fits[card] == (byte)'E' && fits[card + 1] == (byte)'N' && fits[card + 2] == (byte)'D' && fits[card + 3] == (byte)' ')
      {
        int data = (card / 2880 + 1) * 2880;
        Assert.True(data + 2 <= fits.Length, "the fixture has no data unit");
        var copy = (byte[])fits.Clone();
        copy[data + 1] ^= 1; // BITPIX 16: the first pixel's low byte
        return copy;
      }
    }
    throw new InvalidDataException("no END card in the fixture header");
  }

  [SkippableTheory]
  [MemberData(nameof(Variants))]
  public void DateVariantFits_ShowsTheDate_AndModelUsesIt(string fixture, string expectedMidUtc, string? nineteenHundredsMidUtc)
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var (artifacts, saveDir) = ReviewSave.Prepare("fits-date-variants-" + Path.GetFileNameWithoutExtension(fixture));
    var source = FixturePath(fixture);
    Assert.True(File.Exists(source), $"fixture missing: {source}");
    // the fixtures share their 4 × 3 ramp with the save's own fits_lt_67p_header_4x3.fits, and
    // FITS imports deduplicate by the raster: an unchanged copy would merge into that texture
    // ("Already imported") and keep its header. One pixel one count off makes it a new image.
    var path = Path.Combine(artifacts, fixture);
    File.WriteAllBytes(path, NudgeFirstPixel(File.ReadAllBytes(source)));

    using var s = AppSession.Launch(Path.Combine(artifacts, "app.log"), diagnosticPanels: false);
    try
    {
      Step("open the review save");
      ReviewSave.Open(s, Path.Combine(saveDir, "aethervk_index.json"));

      Step("delete the save's own billboards");
      DeleteAllBillboards(s);

      Step($"import {fixture} and make it a billboard");
      FitsComparisonSteps.ImportImage(s, path);
      int row = FitsComparisonSteps.CreateBillboard(s, path);
      Assert.Equal(0, row);
      // a new billboard opens over the app's own controls: hide its window
      FitsComparisonSteps.SetVisible(s, row, visible: false);

      Step("the key facts show the mid-exposure date");
      s.Expand("BillboardFitsSection", "FitsFact_DateMid");
      var dateMid = Ui.WaitFor(() => s.App.FindElementsByAccessibilityId("FitsFact_DateMid").ElementAtOrDefault(row),
        TimeSpan.FromSeconds(20), "the mid-exposure fact");
      output.WriteLine("Mid-exposure UTC: " + dateMid.Text);
      Assert.StartsWith(expectedMidUtc, dateMid.Text, StringComparison.Ordinal);
      s.App.SaveScreenshot(Path.Combine(artifacts, "01-facts.png"));

      if (nineteenHundredsMidUtc is { } earlier)
      {
        Step("the two-digit-year selector: 20xx by default, 19xx moves the date back a century, 20xx back again");
        var century = Ui.WaitFor(() => s.App.FindElementsByAccessibilityId("FitsTwoDigitYearCentury").ElementAtOrDefault(row),
          TimeSpan.FromSeconds(10), "the century selector");
        Assert.StartsWith("20xx", Ui.ItemText(century), StringComparison.Ordinal);
        s.ScrollIntoView(century);
        s.Choose("FitsTwoDigitYearCentury", t => t.StartsWith("19xx", StringComparison.Ordinal));
        Ui.WaitUntil(() => (s.App.FindElementsByAccessibilityId("FitsFact_DateMid").ElementAtOrDefault(row)?.Text ?? "")
          .StartsWith(earlier, StringComparison.Ordinal), TimeSpan.FromSeconds(10), "the 19xx date");
        output.WriteLine("Mid-exposure UTC at 19xx: " + s.App.FindElementsByAccessibilityId("FitsFact_DateMid").ElementAt(row).Text);
        s.App.SaveScreenshot(Path.Combine(artifacts, "01b-facts-19xx.png"));
        s.Choose("FitsTwoDigitYearCentury", t => t.StartsWith("20xx", StringComparison.Ordinal));
        Ui.WaitUntil(() => (s.App.FindElementsByAccessibilityId("FitsFact_DateMid").ElementAtOrDefault(row)?.Text ?? "")
          .StartsWith(expectedMidUtc, StringComparison.Ordinal), TimeSpan.FromSeconds(10), "the 20xx date again");
      }
      else
      {
        Step("a four-digit year: no century selector");
        Assert.True(s.TryFind("FitsTwoDigitYearCentury") is null or { Displayed: false }, "the selector is shown only for a two-digit year");
      }

      Step("Model reads the date: no 'no FITS' / 'no readable date' hint");
      Ui.WaitUntil(() => s.Find("ModeBarModelButton").Enabled, TimeSpan.FromSeconds(60), "Model enabled");
      s.Click("ModeBarModelButton");
      // the hint is cleared at once and later carries the run's status; it never names a
      // missing image or date
      for (int i = 0; i < 10; i++)
      {
        Thread.Sleep(500);
        var hint = s.TryFind("ModeBarHintText")?.Text ?? string.Empty;
        foreach (var bad in NoImageHints)
          Assert.DoesNotContain(bad, hint, StringComparison.Ordinal);
      }
      output.WriteLine("hint after Model: " + (s.TryFind("ModeBarHintText")?.Text ?? "(none)"));
      s.App.SaveScreenshot(Path.Combine(artifacts, "02-model.png"));
    }
    catch
    {
      s.Desktop.SaveScreenshot(Path.Combine(artifacts, "failure.png"));
      throw;
    }
  }

  /// <summary>Clicks every billboard row's Delete button until no row is left.</summary>
  private static void DeleteAllBillboards(AppSession s)
  {
    s.SelectTab("SettingsTab");
    s.Expand("BillboardsSection", "CreateBillboardButton");
    for (int guard = 0; guard < 8; guard++)
    {
      int rows = s.App.FindElementsByAccessibilityId("BillboardGeometry").Count;
      if (rows == 0)
        return;
      var delete = Ui.WaitFor(() => s.App.FindElementsByName("Delete").FirstOrDefault(e => e.Displayed && e.Enabled),
        TimeSpan.FromSeconds(10), "a billboard's Delete button");
      s.ScrollIntoView(delete).RealClick();
      Ui.WaitUntil(() => s.App.FindElementsByAccessibilityId("BillboardGeometry").Count < rows,
        TimeSpan.FromSeconds(10), "the billboard row to go");
    }
    Assert.Empty(s.App.FindElementsByAccessibilityId("BillboardGeometry"));
  }
}
