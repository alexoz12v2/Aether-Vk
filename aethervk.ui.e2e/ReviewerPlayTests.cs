using System.Globalization;
using Xunit.Abstractions;

namespace AetherVk.UI.E2E;

/// <summary>
/// The reviewer's mail of 2026-10-29, "the animation doesn't work, the model only appears if you go
/// straight to the end": the reviewer's 67P save seen from the image's observatory (Set up
/// comparison), the image hidden, Low particle quality (a GTX 1070 class GPU), then Play. While it
/// plays the epoch advances and the viewport changes frame to frame; paused it holds still.
/// Screenshots go to TestResults/e2e/review-play/.
/// </summary>
public sealed class ReviewerPlayTests(ITestOutputHelper output)
{
  private const int Frames = 5;
  private static readonly TimeSpan FrameGap = TimeSpan.FromMilliseconds(1500);

  [SkippableFact]
  public void ReviewerSave_PlaysVisiblyFromTheImageSite()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var artifacts = Path.Combine(AppSession.RepoRoot(), "TestResults", "e2e", "review-play");
    if (Directory.Exists(artifacts))
      Directory.Delete(artifacts, recursive: true);
    var saveDir = Path.Combine(artifacts, "review_67p_2021-12");
    ReviewerMailTests.CopyDir(Path.Combine(AppSession.RepoRoot(), "test_assets", "saves", "review_67p_2021-12"), saveDir);

    // the quality slider writes the preferences: put them back afterwards
    var prefs = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aethervk", "preferences.json");
    var prefsBackup = File.Exists(prefs) ? File.ReadAllBytes(prefs) : null;
    try
    {
      using var s = AppSession.Launch(Path.Combine(artifacts, "app.log"), diagnosticPanels: false);
      try
      {
        Step("open the reviewer's save");
        Thread.Sleep(2000);
        s.Menu("File", "Open...");
        Thread.Sleep(1000);
        s.AnswerFileDialog(Path.Combine(saveDir, "aethervk_index.json"));
        Ui.WaitUntil(() => s.WindowTitle.Contains("review_67p", StringComparison.OrdinalIgnoreCase),
          TimeSpan.FromSeconds(90), "the save to open");
        Thread.Sleep(5000);

        Step("Low particle quality");
        s.SelectTab("SettingsTab");
        s.Expand("ParticleRenderingSection", "ParticleQualitySlider");
        s.SetRange("ParticleQualitySlider", 2, step: 1, tolerance: 0);
        Ui.WaitUntil(() => s.Find("ParticleQualityLabel").Text.Contains("Low", StringComparison.Ordinal),
          TimeSpan.FromSeconds(10), "the Low preset");

        Step("the Earth observer at the image's observatory, the image hidden");
        FitsComparisonSteps.SetUpComparison(s);
        FitsComparisonSteps.SetVisible(s, 0, visible: false);
        Thread.Sleep(5000);
        s.App.SaveScreenshot(Path.Combine(artifacts, "00-paused.png"));

        Step("play at 1 h/s");
        s.SelectTab("TimelineTab");
        s.ChooseAt("SpeedPicker", 2);
        var vp = s.Find("ViewportHost");
        var epochBefore = Epoch(s);
        s.Click("PlayPauseButton");
        Thread.Sleep(2000);
        var playing = new List<double>();
        var previous = Sample(vp);
        for (int i = 1; i <= Frames; i++)
        {
          Thread.Sleep(FrameGap);
          var frame = Sample(vp);
          playing.Add(Changed(previous, frame));
          previous = frame;
          s.App.SaveScreenshot(Path.Combine(artifacts, $"0{i}-playing.png"));
        }
        s.Click("PlayPauseButton");
        var epochAfter = Epoch(s);
        output.WriteLine($"epoch {epochBefore:u} -> {epochAfter:u}; changed while playing: {string.Join(", ", playing.Select(c => c.ToString("P1", CultureInfo.InvariantCulture)))}");
        Assert.True(epochAfter > epochBefore + TimeSpan.FromHours(2), $"the epoch did not advance ({epochBefore:u} -> {epochAfter:u})");
        Assert.All(playing, c => Assert.True(c > 0.02, $"a frame pair while playing barely changed ({c:P1})"));

        Step("paused it holds still");
        Thread.Sleep(3000);
        var a = Sample(vp);
        Thread.Sleep(FrameGap);
        double paused = Changed(a, Sample(vp));
        output.WriteLine($"changed while paused: {paused:P1}");
        Assert.True(paused < playing.Average() / 2, $"paused changed {paused:P1}, as much as playing ({playing.Average():P1})");
      }
      catch
      {
        s.Desktop.SaveScreenshot(Path.Combine(artifacts, "failure.png"));
        throw;
      }
    }
    finally
    {
      if (prefsBackup is not null)
        File.WriteAllBytes(prefs, prefsBackup);
    }
  }

  private static DateTime Epoch(AppSession s) =>
    DateTime.ParseExact(s.Find("CurrentEpochBox").Text.Trim()[..19], "yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);

  /// <summary>The viewport's central 90 %, on a 32 × 32 luminance grid (the nucleus stays centred).</summary>
  private static double[] Sample(OpenQA.Selenium.Appium.AppiumWebElement vp)
  {
    int w = (int)(vp.Size.Width * 0.9), h = (int)(vp.Size.Height * 0.9);
    return Win32.SampleLuminance(vp.Location.X + (vp.Size.Width - w) / 2, vp.Location.Y + (vp.Size.Height - h) / 2, w, h, 32);
  }

  /// <summary>The share of samples whose luminance moved by more than 0.01.</summary>
  private static double Changed(double[] a, double[] b) => a.Zip(b, (x, y) => Math.Abs(x - y) > 0.01 ? 1.0 : 0.0).Average();

  private void Step(string name) => output.WriteLine($"{DateTime.Now:HH:mm:ss} {name}");
}
