using Xunit.Abstractions;

namespace AetherVk.UI.E2E;

/// <summary>
/// The main window's top mode bar (Orbit / Nucleus / Model) drives the same camera mode as the
/// viewport overlay's buttons: after each click the viewport's mode (the Settings tab's
/// "Mode: ..." line, the overlay's mode) follows. Runs on the reviewer's 67P save so a comet is
/// committed and every mode is available.
/// </summary>
public sealed class TopModeBarTests(ITestOutputHelper output)
{
  [SkippableFact]
  public void TopModeBar_SwitchesTheViewMode()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var artifacts = Path.Combine(AppSession.RepoRoot(), "TestResults", "e2e", "top-mode-bar");
    if (Directory.Exists(artifacts))
      Directory.Delete(artifacts, recursive: true);
    var saveDir = Path.Combine(artifacts, "review_67p_2021-12");
    ReviewerMailTests.CopyDir(Path.Combine(AppSession.RepoRoot(), "test_assets", "saves", "review_67p_2021-12"), saveDir);

    using var s = AppSession.Launch(Path.Combine(artifacts, "app.log"), diagnosticPanels: false);
    try
    {
      Thread.Sleep(2000);
      s.Menu("File", "Open...");
      Thread.Sleep(1000);
      s.AnswerFileDialog(Path.Combine(saveDir, "aethervk_index.json"));
      Ui.WaitUntil(() => s.WindowTitle.Contains("review_67p", StringComparison.OrdinalIgnoreCase),
        TimeSpan.FromSeconds(90), "the save to open");
      Thread.Sleep(3000);

      // Nucleus first: the save may already be in Orbit, so Orbit last proves a change back
      foreach (var (button, mode) in new[]
               {
                 ("ModeBarNucleusButton", "CometOrbiting"),
                 ("ModeBarModelButton", "EarthPosition"),
                 ("ModeBarOrbitButton", "UpZenith"),
               })
      {
        output.WriteLine($"{DateTime.Now:HH:mm:ss} {button} -> {mode}");
        Ui.WaitUntil(() => s.Find(button).Enabled, TimeSpan.FromSeconds(60), $"{button} enabled");
        s.Click(button);
        s.SelectTab("SettingsTab");
        s.Expand("ViewportSettingsSection", "CameraModeText");
        Ui.WaitUntil(() => s.Find("CameraModeText").Text.Contains(mode, StringComparison.Ordinal),
          TimeSpan.FromSeconds(30), $"camera mode {mode} after {button}");
        s.App.SaveScreenshot(Path.Combine(artifacts, $"{button}.png"));
      }
    }
    catch
    {
      s.Desktop.SaveScreenshot(Path.Combine(artifacts, "failure.png"));
      throw;
    }
  }
}
