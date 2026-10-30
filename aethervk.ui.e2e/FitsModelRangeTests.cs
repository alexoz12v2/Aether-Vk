using System.Globalization;
using Xunit.Abstractions;

namespace AetherVk.UI.E2E;

/// <summary>
/// The FITS row's "Model range" on the reviewer's 67P save: the image's mid-exposure epoch goes to
/// the Comet tab's "rotations ending at" box (Simulated range), and Start/End become N nucleus rotations ending
/// there (at least 1 day). The save is committed, so the edit is locked until Restore.
/// </summary>
public sealed class FitsModelRangeTests(ITestOutputHelper output)
{
  /// <summary>The committed SPK's spin rate in the fixture (°/day): a 12.41 h period.</summary>
  private const double RateDegDay = 696.2127316680096;

  [SkippableFact]
  public void FitsImage_SetsTheModelRange()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var artifacts = Path.Combine(AppSession.RepoRoot(), "TestResults", "e2e", "fits-model-range");
    if (Directory.Exists(artifacts))
      Directory.Delete(artifacts, recursive: true);
    var saveDir = Path.Combine(artifacts, "review_67p_2021-12");
    ReviewerMailTests.CopyDir(Path.Combine(AppSession.RepoRoot(), "test_assets", "saves", "review_67p_2021-12"), saveDir);
    var imageEpoch = DateTime.ParseExact(FitsComparisonSteps.DateMidUtc, "yyyy-MM-ddTHH:mm:ss", CultureInfo.InvariantCulture);
    double periodH = 360.0 * 24.0 / RateDegDay;

    using var s = AppSession.Launch(Path.Combine(artifacts, "app.log"), diagnosticPanels: false);
    try
    {
      s.Menu("File", "Open...");
      Thread.Sleep(1000);
      s.AnswerFileDialog(Path.Combine(saveDir, "aethervk_index.json"));
      Ui.WaitUntil(() => s.WindowTitle.Contains("review_67p", StringComparison.OrdinalIgnoreCase),
        TimeSpan.FromSeconds(90), "the save to open");
      Thread.Sleep(3000);

      // N = 2: 24.8 h, above the 1-day floor
      s.OpenRangeEntry("ModelRotationsBox");
      s.TypeInto(s.Find("ModelRotationsBox"), "2", enter: false);

      s.SelectTab("SettingsTab");
      s.Expand("BillboardsSection", "CreateBillboardButton");
      s.Expand("BillboardFitsSection", "BillboardFitsModelRange");
      s.ClickWhenSettled("BillboardFitsModelRange");

      s.OpenRangeEntry("ModelEndEpochBox");
      Ui.WaitUntil(() => s.Find("ModelEndEpochBox").Text.StartsWith(FitsComparisonSteps.DateMidUtc, StringComparison.Ordinal),
        TimeSpan.FromSeconds(10), "the image epoch in the model end box");
      AssertRange(s, imageEpoch, TimeSpan.FromHours(2 * periodH), "2 rotations");
      Assert.Contains("Range locked", s.App.PageSource, StringComparison.Ordinal);
      s.App.SaveScreenshot(Path.Combine(artifacts, "01-model-range.png"));

      // N = 1: 12.4 h, the 1-day floor wins
      s.TypeInto(s.Find("ModelRotationsBox"), "1", enter: false);
      s.Click("ApplyModelRangeButton");
      AssertRange(s, imageEpoch, TimeSpan.FromDays(1), "1 rotation (1-day floor)");

      // Restore: the committed range back
      s.Click("RestoreButton");
      Ui.WaitUntil(() => ReviewSave.Shows(s.Find("StartEpochBox").Text, ReviewSave.CommittedStartUtc)
                         && ReviewSave.Shows(s.Find("EndEpochBox").Text, ReviewSave.CommittedEndUtc),
        TimeSpan.FromSeconds(10), "Restore to put the committed range back");
      Assert.DoesNotContain("Range locked", s.App.PageSource, StringComparison.Ordinal);
    }
    catch
    {
      s.Desktop.SaveScreenshot(Path.Combine(artifacts, "failure.png"));
      throw;
    }
  }

  private void AssertRange(AppSession s, DateTime end, TimeSpan span, string what)
  {
    Ui.WaitUntil(() => Parse(s.Find("EndEpochBox").Text) == end, TimeSpan.FromSeconds(10), $"{what}: the end at the image epoch");
    var start = Parse(s.Find("StartEpochBox").Text);
    output.WriteLine($"{what}: {start:s} -> {end:s}");
    Assert.True(Math.Abs((end - start - span).TotalSeconds) <= 1, $"{what}: span {end - start}, expected {span}");
  }

  private static DateTime Parse(string text) =>
    DateTime.ParseExact(text.Trim().TrimEnd('Z')[..19], "yyyy-MM-ddTHH:mm:ss", CultureInfo.InvariantCulture);
}
