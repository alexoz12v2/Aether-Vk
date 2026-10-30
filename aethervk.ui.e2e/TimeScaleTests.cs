using System.Globalization;
using Xunit.Abstractions;

namespace AetherVk.UI.E2E;

/// <summary>
/// Dates are UTC by default (Settings › Time scale for dates): on the reviewer's 67P save the
/// image's DATE-OBS mid-exposure typed into the Timeline tab's epoch box seeks there, the box
/// reads back the same UTC (not 69 s off), the labels say "(UTC)", and the FITS facts show the
/// mid-exposure in both scales, 69.18 s apart.
/// </summary>
public sealed class TimeScaleTests(ITestOutputHelper output)
{
  [SkippableFact]
  public void TypedUtcEpoch_ReadsBackAsUtc_AndFitsFactsShowBothScales()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var artifacts = Path.Combine(AppSession.RepoRoot(), "TestResults", "e2e", "time-scale");
    if (Directory.Exists(artifacts))
      Directory.Delete(artifacts, recursive: true);
    var saveDir = Path.Combine(artifacts, "review_67p_2021-12");
    ReviewerMailTests.CopyDir(Path.Combine(AppSession.RepoRoot(), "test_assets", "saves", "review_67p_2021-12"), saveDir);
    var midUtc = DateTime.ParseExact(FitsComparisonSteps.DateMidUtc, "yyyy-MM-ddTHH:mm:ss", CultureInfo.InvariantCulture);

    using var s = AppSession.Launch(Path.Combine(artifacts, "app.log"), diagnosticPanels: false);
    try
    {
      s.Menu("File", "Open...");
      Thread.Sleep(1000);
      s.AnswerFileDialog(Path.Combine(saveDir, "aethervk_index.json"));
      Ui.WaitUntil(() => s.WindowTitle.Contains("review_67p", StringComparison.OrdinalIgnoreCase),
        TimeSpan.FromSeconds(90), "the save to open");
      Thread.Sleep(3000);

      // the Timeline tab: the epoch box in UTC
      s.SelectTab("TimelineTab");
      Assert.Equal("Epoch (UTC)", s.Find("CurrentEpochLabel").Text.Trim());
      s.TypeInto(s.Find("CurrentEpochBox"), midUtc.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture));
      Ui.WaitUntil(() => Epoch(s) == midUtc, TimeSpan.FromSeconds(10), "the epoch box to read back the typed UTC");
      output.WriteLine($"epoch box: {s.Find("CurrentEpochBox").Text}");
      s.App.SaveScreenshot(Path.Combine(artifacts, "01-epoch-utc.png"));

      // the FITS facts: mid-exposure in UTC and TDB
      s.SelectTab("SettingsTab");
      s.Expand("BillboardsSection", "CreateBillboardButton");
      s.Expand("BillboardFitsSection", "FitsFact_DateMid");
      var utcText = s.Find("FitsFact_DateMid").Text;
      var tdbText = s.Find("FitsFact_DateMidTdb").Text;
      output.WriteLine($"mid-exposure: {utcText} / {tdbText}");
      Assert.StartsWith(FitsComparisonSteps.DateMidUtc, utcText, StringComparison.Ordinal);
      Assert.Contains("TDB", tdbText, StringComparison.Ordinal);
      var tdb = DateTime.ParseExact(tdbText.Trim()[..19], "yyyy-MM-ddTHH:mm:ss", CultureInfo.InvariantCulture);
      Assert.InRange((tdb - midUtc).TotalSeconds, 68, 70);
      s.App.SaveScreenshot(Path.Combine(artifacts, "02-fits-facts.png"));
    }
    catch
    {
      s.Desktop.SaveScreenshot(Path.Combine(artifacts, "failure.png"));
      throw;
    }
  }

  private static DateTime Epoch(AppSession s) =>
    DateTime.ParseExact(s.Find("CurrentEpochBox").Text.Trim()[..19], "yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);
}
