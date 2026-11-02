using System.Globalization;
using Xunit.Abstractions;

namespace AetherVk.UI.E2E;

/// <summary>
/// The slow-rotator fixture <c>test_assets/saves/review_slow_rotator_2026-11.avk</c> (UN271, 60 km
/// nucleus, 2 jets) opens offline, shows its nucleus and both jets, plays, and the app then closes
/// cleanly: exit code 0, no "Fatal error" or "invalid camera entity" in its log.
/// Artifacts go to TestResults/e2e/slow-rotator/.
/// </summary>
public sealed class SlowRotatorFixtureTests(ITestOutputHelper output)
{
  [SkippableFact]
  public void SlowRotatorFixture_OpensPlaysAndClosesCleanly()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var root = AppSession.RepoRoot();
    var artifacts = Path.Combine(root, "TestResults", "e2e", "slow-rotator");
    if (Directory.Exists(artifacts))
      Directory.Delete(artifacts, recursive: true);
    Directory.CreateDirectory(artifacts);
    var avk = Path.Combine(artifacts, "review_slow_rotator_2026-11.avk");
    File.Copy(Path.Combine(root, "test_assets", "saves", "review_slow_rotator_2026-11.avk"), avk);
    var log = Path.Combine(artifacts, "app.log");

    var s = AppSession.Launch(log, diagnosticPanels: false);
    int? exitCode = null;
    try
    {
      Step("open the fixture");
      Thread.Sleep(2000);
      s.Menu("File", "Open...");
      Thread.Sleep(1000);
      s.AnswerFileDialog(avk);
      Ui.WaitUntil(() => s.WindowTitle.Contains("slow_rotator", StringComparison.OrdinalIgnoreCase),
        TimeSpan.FromSeconds(90), "the fixture to open");
      Thread.Sleep(10000);

      Step("the 60 km nucleus and both jets");
      s.SelectTab("CometTab");
      s.Expand("NucleusSection", "NucleusRadiusSlider");
      double radius = s.ReadSlider("NucleusRadiusSlider");
      output.WriteLine($"nucleus radius {radius.ToString(CultureInfo.InvariantCulture)} km");
      Assert.InRange(radius, 59.5, 60.5);
      s.SelectTab("ModelTab");
      s.Expand("DustSection", "AddJetButton");
      Ui.WaitUntil(() => s.Items("JetListBox").Count == 2, TimeSpan.FromSeconds(60), "2 jet rows");
      Thread.Sleep(5000);
      Assert.DoesNotContain("Not in the simulation", s.App.PageSource);
      s.App.SaveScreenshot(Path.Combine(artifacts, "00-opened.png"));

      Step("play at 1 h/s");
      s.SelectTab("TimelineTab");
      s.ChooseAt("SpeedPicker", ReviewSave.OneHourPerSecondIndex);
      // the save opens on its observation epoch, which is the committed end: seek back to play
      s.TypeInto(s.Find("CurrentEpochBox"), "2026-10-01 00:00:00", enter: false);
      s.Click("SeekToEpochButton");
      Ui.WaitUntil(() => Epoch(s) < new DateTime(2026, 10, 2), TimeSpan.FromSeconds(30), "the seek to 2026-10-01");
      Thread.Sleep(10000);
      var vp = s.Find("ViewportHost");
      var epochBefore = Epoch(s);
      s.Click("PlayPauseButton");
      Thread.Sleep(2000);
      var changes = new List<double>();
      var previous = FrameMetrics.Sample(vp);
      for (int i = 1; i <= 4; i++)
      {
        Thread.Sleep(1500);
        var frame = FrameMetrics.Sample(vp);
        changes.Add(FrameMetrics.Changed(previous, frame));
        previous = frame;
        s.App.SaveScreenshot(Path.Combine(artifacts, $"0{i}-playing.png"));
      }
      s.Click("PlayPauseButton");
      var epochAfter = Epoch(s);
      output.WriteLine($"epoch {epochBefore:u} -> {epochAfter:u}; changed while playing: " +
        string.Join(", ", changes.Select(c => c.ToString("P1", CultureInfo.InvariantCulture))));
      Assert.True(epochAfter > epochBefore + TimeSpan.FromHours(2), $"the epoch did not advance ({epochBefore:u} -> {epochAfter:u})");
      Thread.Sleep(3000);
      s.App.SaveScreenshot(Path.Combine(artifacts, "05-paused.png"));

      Step("close");
      s.Process.CloseMainWindow();
      if (s.Process.WaitForExit(30_000))
        exitCode = s.Process.ExitCode;
    }
    catch
    {
      try { s.Desktop.SaveScreenshot(Path.Combine(artifacts, "failure.png")); } catch { }
      throw;
    }
    finally
    {
      s.Dispose();
    }

    output.WriteLine($"exit code {exitCode?.ToString(CultureInfo.InvariantCulture) ?? "none (killed after 30 s)"}");
    Assert.Equal(0, exitCode);
    var text = File.ReadAllText(log);
    foreach (var bad in new[] { "Fatal error", "invalid camera entity" })
      Assert.False(text.Contains(bad, StringComparison.OrdinalIgnoreCase), $"the app log has \"{bad}\"");
  }

  private static DateTime Epoch(AppSession s) =>
    DateTime.ParseExact(s.Find("CurrentEpochBox").Text.Trim()[..19], "yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);

  private void Step(string name) => output.WriteLine($"{DateTime.Now:HH:mm:ss} {name}");
}
