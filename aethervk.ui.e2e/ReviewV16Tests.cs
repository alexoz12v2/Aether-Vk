using System.Globalization;
using System.Text.RegularExpressions;
using OpenQA.Selenium;
using Xunit.Abstractions;

namespace AetherVk.UI.E2E;

/// <summary>
/// The v1.0.1-16 checks (HANDOFF_windows_2026-11-03 §5): both committed saves, dumped by v1.0.1-15
/// (scene dump v9), reopen with their saved state restored, not sought; the Model tab's new jet
/// controls; a click-opened help flyout stays up; the ephemeris panel's two Sun position-angle rows.
/// Artifacts go to TestResults/e2e/review-v16/.
/// </summary>
public sealed partial class ReviewV16Tests(ITestOutputHelper output)
{
  [SkippableFact]
  public void SlowRotatorSave_RestoresItsState() =>
    Run("review_slow_rotator_2026-11.avk", "slow_rotator", "slow-rotator", _ => { });

  [SkippableFact]
  public void Review67pSave_RestoresItsState_NewJetControls_PinnedHelp_TwoSunPaRows() =>
    Run("review_67p_2021-12.avk", "review_67p", "67p", s =>
    {
      Step("the ephemeris panel's two Sun position angles");
      var pa = Ui.WaitFor(() => ReviewSave.OverlayText(s, "EphemerisPaOfSun") is { Length: > 0 } t ? t : null,
        TimeSpan.FromSeconds(20), "the PA of Sun row");
      var psAng = Ui.WaitFor(() => ReviewSave.OverlayText(s, "EphemerisPsAng") is { Length: > 0 } t ? t : null,
        TimeSpan.FromSeconds(20), "the PsAng row");
      output.WriteLine($"PA of Sun {pa}; PsAng {psAng}");
      double diff = ((Degrees(pa) - Degrees(psAng)) % 360 + 360) % 360;
      Assert.InRange(diff, 179.8, 180.2);

      Step("a click-opened help flyout stays up");
      s.SelectTab("TimelineTab");
      var before = Win32.TopLevelWindows(s.Process.Id).Select(w => w.Handle).ToHashSet();
      s.ClickWhenSettled("SpeedPickerHelp");
      Thread.Sleep(300);
      bool PopupUp() => Win32.TopLevelWindows(s.Process.Id)
        .Any(w => !before.Contains(w.Handle) && Win32.FrameBounds(w.Handle) is { Width: >= 30, Height: >= 20 });
      Ui.WaitUntil(PopupUp, TimeSpan.FromSeconds(3), "the help flyout");
      ParkPointer();
      Thread.Sleep(1500);
      s.App.SaveScreenshot(Path.Combine(_artifacts!, "flyout-after-1.5s.png"));
      Assert.True(PopupUp(), "the click-opened help flyout closed within 1.5 s");
      Win32.PressKey(Win32.VK_ESCAPE);
      Thread.Sleep(500);

      Step("the Model tab: defaults q box, β range, dynamic model");
      s.SelectTab("ModelTab");
      s.Expand("DustSection", "AddJetButton");
      Assert.NotNull(s.TryFind("DustSizeQBox"));
      Assert.Null(s.TryFind("SizeDistributionAdvancedExpander"));
      Ui.WaitUntil(() => s.Items("JetListBox").Count > 0, TimeSpan.FromSeconds(60), "the jet rows");
      if (s.TryFind("JetLatitudeSlider") is null)
      {
        s.ScrollIntoView(s.Items("JetListBox")[0]).RealClick();
        Ui.WaitFor(() => s.TryFind("JetLatitudeSlider"), TimeSpan.FromSeconds(10), "the jet detail");
      }
      var beta = s.Find("JetBetaRangeText").Text;
      output.WriteLine($"β range: {beta}");
      Assert.Matches(@"\d.*\s[–…-]\s.*\d", beta); // "β/F – β·F"

      s.Expand("JetDynamicsExpander", "JetInsolationWeighted");
      foreach (var id in new[] { "JetInsolationWeighted", "JetOrbitSizeLaw", "JetSpkBeforeStart", "JetLifetimeDays", "JetEmissionStepsText" })
        Assert.NotNull(s.TryFind(id));
      output.WriteLine($"lifetime {s.Find("JetLifetimeDays").Text} d, emission steps {s.Find("JetEmissionStepsText").Text}");
      Assert.False(string.IsNullOrWhiteSpace(s.Find("JetEmissionStepsText").Text), "no emission steps shown");

      Step("the q table: on with the orbit size law, a row added");
      if (s.TryFind("JetSizeQTableAddRow") is null)
        s.Click("JetOrbitSizeLaw");
      s.ScrollIntoView(Ui.WaitFor(() => s.TryFind("JetSizeQTableAddRow"), TimeSpan.FromSeconds(10), "the q table"));
      Thread.Sleep(500);
      // JetSizeQCurvePlot is not checked: CurvePlot has no automation peer, so no plot is ever in
      // the UIA tree (the v1.0.1-15 plots neither); the screenshot below shows it
      int Rows() => s.Find("JetSizeQTable").FindElementsByClassName("TextBox").Count / 2;
      int rows = Rows();
      s.Click("JetSizeQTableAddRow");
      Ui.WaitUntil(() => Rows() == rows + 1, TimeSpan.FromSeconds(10), $"{rows + 1} q rows");
      output.WriteLine($"q rows {rows} -> {Rows()}");
      s.App.SaveScreenshot(Path.Combine(_artifacts!, "model-dynamics.png"));
    });

  private string? _artifacts;

  private void Run(string save, string titlePart, string name, Action<AppSession> checks)
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var root = AppSession.RepoRoot();
    _artifacts = Path.Combine(root, "TestResults", "e2e", "review-v16", name);
    if (Directory.Exists(_artifacts))
      Directory.Delete(_artifacts, recursive: true);
    Directory.CreateDirectory(_artifacts);
    var avk = Path.Combine(_artifacts, save);
    File.Copy(Path.Combine(root, "test_assets", "saves", save), avk);
    var log = Path.Combine(_artifacts, "app.log");

    var s = AppSession.Launch(log, diagnosticPanels: false);
    int? exitCode = null;
    try
    {
      Step($"open {save}");
      Thread.Sleep(2000);
      s.Menu("File", "Open...");
      Thread.Sleep(1000);
      s.AnswerFileDialog(avk);
      Ui.WaitUntil(() => s.WindowTitle.Contains(titlePart, StringComparison.OrdinalIgnoreCase),
        TimeSpan.FromSeconds(90), "the save to open");
      var opened = Ui.WaitFor(() => OpenLine(log), TimeSpan.FromSeconds(120), "the [Open] log line");
      output.WriteLine(opened);
      Thread.Sleep(8000);
      s.App.SaveScreenshot(Path.Combine(_artifacts, "00-opened.png"));
      Assert.Contains("state restored: True", opened, StringComparison.Ordinal);
      Assert.Contains("sought: False", opened, StringComparison.Ordinal);
      Assert.DoesNotContain("[Seek] applied", ReadShared(log), StringComparison.Ordinal);

      checks(s);

      Step("close");
      s.Process.CloseMainWindow();
      // the checks edit the scene (the size law, a q row): the save prompt asks first
      if (!s.Process.WaitForExit(5_000))
        try
        {
          Ui.WaitFor(() => s.Desktop.FindElementsByAccessibilityId("Discard").FirstOrDefault(e => e.Displayed),
            TimeSpan.FromSeconds(10), "the save prompt").Click();
        }
        catch (Exception e) when (e is TimeoutException or WebDriverException) { }
      if (s.Process.WaitForExit(30_000))
        exitCode = s.Process.ExitCode;
    }
    catch
    {
      try { s.Desktop.SaveScreenshot(Path.Combine(_artifacts, "failure.png")); } catch { }
      throw;
    }
    finally
    {
      s.Dispose();
    }

    output.WriteLine($"exit code {exitCode?.ToString(CultureInfo.InvariantCulture) ?? "none (killed after 30 s)"}");
    Assert.Equal(0, exitCode);
    var text = ReadShared(log);
    foreach (var bad in new[] { "Fatal error", "invalid camera entity" })
      Assert.False(text.Contains(bad, StringComparison.OrdinalIgnoreCase), $"the app log has \"{bad}\"");
  }

  private static string? OpenLine(string log) =>
    ReadShared(log).Split('\n').FirstOrDefault(l => l.Contains("[Open] state restored", StringComparison.Ordinal))?.Trim();

  /// <summary>The log while the app still writes it.</summary>
  private static string ReadShared(string path)
  {
    if (!File.Exists(path))
      return "";
    using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
    using var reader = new StreamReader(stream);
    return reader.ReadToEnd();
  }

  private static double Degrees(string text) =>
    double.Parse(Number().Match(text.Replace('−', '-')).Value, CultureInfo.InvariantCulture);

  [GeneratedRegex(@"-?\d+(\.\d+)?")]
  private static partial Regex Number();

  /// <summary>The pointer far from the app (the other monitor, or the top edge of this one).</summary>
  private static void ParkPointer()
  {
    var monitors = Win32.Monitors();
    var m = monitors.Skip(1).FirstOrDefault() ?? monitors[0];
    Win32.MoveCursor(m.Left + m.Width / 2, m.Top + (monitors.Count > 1 ? m.Height / 2 : 1));
  }

  private void Step(string text) => output.WriteLine($"{DateTime.Now:HH:mm:ss} {text}");
}
