using System.Globalization;
using Xunit.Abstractions;

namespace AetherVk.UI.E2E;

/// <summary>
/// The reviewer's mail of 2026-10-29, item by item, on the reviewer's own 67P save
/// (<c>test_assets/saves/review_67p_2021-12</c>, four jets): every jet in the simulation, the
/// timeline showing the committed range, dust from the first frame, no diagnostic panels by
/// default, no Afρ, a Q_d per jet, the 1/ρ and trajectory view options, the ruler in km and a
/// single-file save that reopens. Screenshots for a by-eye check go to TestResults/e2e/review-mail/.
/// </summary>
public sealed class ReviewerMailTests(ITestOutputHelper output)
{
  [SkippableFact]
  public void ReviewerSave_AnswersTheMail()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var artifacts = Path.Combine(AppSession.RepoRoot(), "TestResults", "e2e", "review-mail");
    if (Directory.Exists(artifacts))
      Directory.Delete(artifacts, recursive: true);
    var saveDir = Path.Combine(artifacts, "review_67p_2021-12");
    CopyDir(Path.Combine(AppSession.RepoRoot(), "test_assets", "saves", "review_67p_2021-12"), saveDir);
    var avk = Path.Combine(artifacts, "review_67p.avk");

    using (var s = AppSession.Launch(Path.Combine(artifacts, "app-open.log"), diagnosticPanels: false))
    {
      try
      {
        Step("open the reviewer's save");
        Thread.Sleep(2000);
        s.Menu("File", "Open...");
        Thread.Sleep(1000);
        s.AnswerFileDialog(Path.Combine(saveDir, "aethervk_index.json"));
        Ui.WaitUntil(() => s.WindowTitle.Contains("review_67p", StringComparison.OrdinalIgnoreCase),
          TimeSpan.FromSeconds(90), "the save to open");

        Step("the FITS textures load");
        Thread.Sleep(5000);
        Assert.DoesNotContain("could not be loaded", s.App.PageSource, StringComparison.OrdinalIgnoreCase);
        // the billboards are their own top-level windows: the placeholder of a failed texture
        Assert.Empty(s.Desktop.FindElementsByName("texture not loaded"));

        Step("R4: no coordinate panels by default");
        Assert.Null(s.Desktop.FindElementsByAccessibilityId("CameraHudCollapseButton").FirstOrDefault(e => e.Displayed));

        Step("R1: four jets, all in the simulation");
        s.SelectTab("ModelTab");
        s.Expand("DustSection", "AddJetButton");
        Ui.WaitUntil(() => s.Items("JetListBox").Count == 4, TimeSpan.FromSeconds(60), "4 jet rows");
        Thread.Sleep(5000); // the native creation of each jet
        Assert.DoesNotContain("Not in the simulation", VisibleText(s));

        Step("R6: no Afρ, a Q_d per jet");
        var source = s.App.PageSource;
        Assert.DoesNotContain("Afρ", source, StringComparison.Ordinal);
        Assert.DoesNotContain("Afrho", source, StringComparison.OrdinalIgnoreCase);
        for (int j = 0; j < 4; j++)
        {
          s.Items("JetListBox")[j].RealClick();
          Thread.Sleep(500);
          s.Expand("JetActivityExpander", "JetDustRate0");
          var qd = s.Find("JetDustRate0").Text;
          output.WriteLine($"jet {j + 1}: Q_d at 1 AU = {qd}");
          Assert.True(double.TryParse(qd, NumberStyles.Float, CultureInfo.InvariantCulture, out var v) && v > 0,
            $"jet {j + 1}: Q_d '{qd}'");
        }

        Step("R2: the timeline shows the committed range");
        s.SelectTab("TimelineTab");
        var start = s.Find("StartEpochBox").Text;
        output.WriteLine($"start box: {start}; committed: {s.TryFind("CommittedRangeText")?.Text}");
        Assert.StartsWith("2021-12-12", start);

        Step("R3: dust from the first frame");
        Thread.Sleep(15000);
        s.App.SaveScreenshot(Path.Combine(artifacts, "01-open-start-epoch.png"));
        s.Click("PlayPauseButton");
        Thread.Sleep(10000);
        s.Click("PlayPauseButton");
        s.App.SaveScreenshot(Path.Combine(artifacts, "02-after-play.png"));

        Step("R8: 1/ρ normalisation, exposure, trajectory lines");
        s.SelectTab("ModelTab");
        s.Expand("DustSection", "DustRhoNormToggle");
        s.Find("DustRhoNormToggle").RealClick();
        Thread.Sleep(4000);
        s.App.SaveScreenshot(Path.Combine(artifacts, "03-rho-norm.png"));
        s.Find("DustTrajectoriesToggle").RealClick();
        Thread.Sleep(4000);
        s.App.SaveScreenshot(Path.Combine(artifacts, "04-trajectories.png"));

        Step("R5: the ruler in km");
        s.Click("ModeModelButton");
        Thread.Sleep(4000);
        s.App.SaveScreenshot(Path.Combine(artifacts, "05-model-view.png"));
        var scale = s.TryFind("EarthObserverScaleText")?.Text ?? string.Empty;
        output.WriteLine($"scale: {scale}");

        Step("R2: editing the committed range locks it, Restore puts it back");
        s.SelectTab("TimelineTab");
        s.TypeInto(s.Find("StartEpochBox"), "2021-12-15T00:00:00", enter: false);
        Ui.WaitUntil(() => s.App.PageSource.Contains("Range locked", StringComparison.Ordinal), TimeSpan.FromSeconds(10), "the lock line");
        s.App.SaveScreenshot(Path.Combine(artifacts, "06-range-locked.png"));
        s.App.FindElementsByName("Restore").First(e => e.Displayed).RealClick();
        Ui.WaitUntil(() => s.Find("StartEpochBox").Text.StartsWith("2021-12-12", StringComparison.Ordinal),
          TimeSpan.FromSeconds(10), "Restore to put the committed start back");

        Step("R9: Save As a single .avk");
        s.Menu("File", "Save As...");
        s.AnswerFileDialog(avk);
        Ui.WaitUntil(() => File.Exists(avk) && !s.WindowTitle.EndsWith('*'), TimeSpan.FromSeconds(90), "the .avk save");
        output.WriteLine($".avk: {new FileInfo(avk).Length} bytes");
      }
      catch
      {
        s.Desktop.SaveScreenshot(Path.Combine(artifacts, "failure-open.png"));
        throw;
      }
    }

    using (var s = AppSession.Launch(Path.Combine(artifacts, "app-reopen.log"), diagnosticPanels: false))
    {
      try
      {
        Step("R9: reopen the .avk");
        Thread.Sleep(2000);
        s.Menu("File", "Open...");
        Thread.Sleep(1000);
        s.AnswerFileDialog(avk);
        Ui.WaitUntil(() => s.WindowTitle.Contains("review_67p", StringComparison.OrdinalIgnoreCase),
          TimeSpan.FromSeconds(90), "the .avk to open");
        s.SelectTab("ModelTab");
        s.Expand("DustSection", "AddJetButton");
        Ui.WaitUntil(() => s.Items("JetListBox").Count == 4, TimeSpan.FromSeconds(60), "4 jet rows after reopen");
        Thread.Sleep(15000);
        Assert.DoesNotContain("Not in the simulation", VisibleText(s));
        s.App.SaveScreenshot(Path.Combine(artifacts, "07-reopened-avk.png"));
      }
      catch
      {
        s.Desktop.SaveScreenshot(Path.Combine(artifacts, "failure-reopen.png"));
        throw;
      }
    }
  }

  private static string VisibleText(AppSession s) => s.App.PageSource;

  private void Step(string name) => output.WriteLine($"{DateTime.Now:HH:mm:ss} {name}");

  internal static void CopyDir(string from, string to)
  {
    Directory.CreateDirectory(to);
    foreach (var f in Directory.GetFiles(from, "*", SearchOption.AllDirectories))
    {
      var dest = Path.Combine(to, Path.GetRelativePath(from, f));
      Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
      File.Copy(f, dest, overwrite: true);
    }
  }
}
