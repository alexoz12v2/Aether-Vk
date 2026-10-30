using System.Diagnostics;
using System.Globalization;
using System.Text.RegularExpressions;
using Xunit.Abstractions;

namespace AetherVk.UI.E2E;

/// <summary>
/// The mail tests' session: the reviewer's save at the default quality, the diagnostic panels
/// off (R4 checks the default), its FITS textures loaded, then its image hidden.
/// </summary>
public sealed class ReviewerMailFixture : ReviewSaveFixture
{
  protected override string ArtifactsName => "review-mail";

  protected override void Opened(AppSession s)
  {
    Assert.DoesNotContain("could not be loaded", s.App.PageSource, StringComparison.OrdinalIgnoreCase);
    // the billboards are their own top-level windows: the placeholder of a failed texture
    Assert.Empty(s.Desktop.FindElementsByName("texture not loaded"));
  }
}

/// <summary>
/// The reviewer's mail of 2026-10-29, one test per item, on the reviewer's own 67P save
/// (<c>test_assets/saves/review_67p_2021-12</c>, four jets): every jet in the simulation, the
/// timeline showing the committed range, dust from the first frame, no diagnostic panels by
/// default, the ruler in km, no Afρ, a Q_d and a size exponent per jet, the 1/ρ and trajectory
/// view options, a single-file save that reopens and a crash that leaves a report. R1–R8 share
/// one session (in name order); R9 and R10 start their own. "The animation doesn't work" is
/// <see cref="PlaybackTests.Play_ChangesTheDustEveryFrame"/> and <see cref="ReviewerPlayTests"/>.
/// Screenshots for a by-eye check go to TestResults/e2e/review-mail/.
/// </summary>
[TestCaseOrderer(NaturalNameOrderer.TypeName, NaturalNameOrderer.AssemblyName)]
public sealed partial class ReviewerMailTests(ReviewerMailFixture fixture, ITestOutputHelper output) : IClassFixture<ReviewerMailFixture>
{
  /// <summary>R3: the dust ring around the comet is at least this much brighter than the sky (luminance 0..1).</summary>
  private const double DustOverSky = 0.02;

  /// <summary>R8: a view option changes more than this share of the grid cells...</summary>
  private const double OptionChangesTheFrame = 0.01;

  /// <summary>...and switching it back leaves less than this share changed from before.</summary>
  private const double OptionRestoresTheFrame = 0.005;

  [SkippableFact]
  public void R1_FourJets_AllInTheSimulation()
  {
    var s = Shared();
    Guard("r1", () =>
    {
      s.SelectTab("ModelTab");
      s.Expand("DustSection", "AddJetButton");
      Ui.WaitUntil(() => s.Items("JetListBox").Count == 4, TimeSpan.FromSeconds(60), "4 jet rows");
      Thread.Sleep(5000); // the native creation of each jet
      Assert.DoesNotContain("Not in the simulation", s.App.PageSource);
    });
  }

  [SkippableFact]
  public void R2_TheTimeline_ShowsTheCommittedRange_AndEditingItLocksIt()
  {
    var s = Shared();
    Guard("r2", () =>
    {
      s.SelectTab("TimelineTab");
      var start = s.Find("TimelineStartEpochBox").Text;
      output.WriteLine($"start box: {start}; committed: {s.TryFind("CommittedRangeText")?.Text}");
      Assert.StartsWith("2021-12-12", start);

      s.OpenRangeEntry();
      s.TypeInto(s.Find("StartEpochBox"), "2021-12-15T00:00:00", enter: false);
      Ui.WaitUntil(() => s.App.PageSource.Contains("Range locked", StringComparison.Ordinal), TimeSpan.FromSeconds(10), "the lock line");
      fixture.Screenshot("r2-range-locked.png");
      s.Click("RestoreButton");
      Ui.WaitUntil(() => s.Find("StartEpochBox").Text.StartsWith("2021-12-12", StringComparison.Ordinal),
        TimeSpan.FromSeconds(10), "Restore to put the committed start back");
    });
  }

  /// <summary>
  /// At the start epoch, before any Play, the ring 10–60 px around the comet is brighter than the
  /// darkest corner of the viewport (the sky) by <see cref="DustOverSky"/>: dust is drawn from
  /// the first frame, not only after a run to the end.
  /// </summary>
  [SkippableFact]
  public void R3_DustIsDrawn_AtTheStartEpoch_BeforePlay()
  {
    var s = Shared();
    Guard("r3", () =>
    {
      ReviewSave.ResetToStart(s);
      Assert.False(ReviewSave.IsPlaying(s));
      var vp = s.Find("ViewportHost");
      (double Ring, double Sky) m = default;
      var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(30);
      do
      {
        Thread.Sleep(1000);
        m = FrameMetrics.RingAndSky(vp, FitsComparisonSteps.CometScreenPx(s));
      } while (m.Ring - m.Sky < DustOverSky && DateTime.UtcNow < deadline);
      fixture.Screenshot("r3-start-epoch.png");
      output.WriteLine($"epoch {ReviewSave.Epoch(s):u}: ring {m.Ring:0.0000}, sky {m.Sky:0.0000}, contrast {m.Ring - m.Sky:0.0000}");
      Assert.False(ReviewSave.IsPlaying(s), "the check must run before Play");
      Assert.True(m.Ring - m.Sky >= DustOverSky,
        $"no dust around the comet at the start epoch (ring {m.Ring:0.0000} vs sky {m.Sky:0.0000})");
    });
  }

  [SkippableFact]
  public void R4_NoCoordinatePanels_ByDefault_TheEphemerisInTheirPlace()
  {
    var s = Shared();
    Guard("r4", () =>
    {
      Assert.Null(s.Desktop.FindElementsByAccessibilityId("CameraHudDiagnostics").FirstOrDefault(e => e.Displayed));
      Ui.WaitUntil(() => ReviewSave.OverlayText(s, "EphemerisR")?.EndsWith(" AU", StringComparison.Ordinal) == true,
        TimeSpan.FromSeconds(10), "the ephemeris r row");
    });
  }

  /// <summary>
  /// The ruler (the overlay's scale bar) in the Model view, the Earth observer: an angle with its
  /// length at the comet ("20 arcsec ≈ 38,300 km at the comet"), or a length in km; never AU.
  /// </summary>
  [SkippableFact]
  public void R5_TheRuler_IsInKm()
  {
    var s = Shared();
    Guard("r5", () =>
    {
      Ui.WaitUntil(() => s.Find("ModeBarModelButton").Enabled, TimeSpan.FromSeconds(60), "the Model button enabled");
      s.Click("ModeBarModelButton");
      Thread.Sleep(4000);
      fixture.Screenshot("r5-model-view.png");
      var ruler = Ui.WaitFor(() => ReviewSave.OverlayText(s, "ViewportRulerText") is { Length: > 0 } t ? t : null,
        TimeSpan.FromSeconds(20), "the ruler label");
      output.WriteLine($"ruler: {ruler}; scale: {s.TryFind("EarthObserverScaleText")?.Text}");
      Assert.Matches(RulerInKm(), ruler);
      Assert.DoesNotContain("AU", ruler, StringComparison.Ordinal);
    });
  }

  [SkippableFact]
  public void R6_NoAfrho_Anywhere()
  {
    var s = Shared();
    Guard("r6", () =>
    {
      s.SelectTab("ModelTab");
      s.Expand("DustSection", "AddJetButton");
      var source = s.App.PageSource;
      Assert.DoesNotContain("Afρ", source, StringComparison.Ordinal);
      Assert.DoesNotContain("Afrho", source, StringComparison.OrdinalIgnoreCase);
    });
  }

  /// <summary>Each jet has a dust production rate Q_d &gt; 0 and a size exponent q within 2–5.</summary>
  [SkippableFact]
  public void R7_EachJet_HasAQdAndASizeExponent()
  {
    var s = Shared();
    Guard("r7", () =>
    {
      s.SelectTab("ModelTab");
      s.Expand("DustSection", "AddJetButton");
      Ui.WaitUntil(() => s.Items("JetListBox").Count == 4, TimeSpan.FromSeconds(60), "4 jet rows");
      for (int j = 0; j < 4; j++)
      {
        s.Items("JetListBox")[j].RealClick();
        Thread.Sleep(500);
        s.Expand("JetActivityExpander", "JetDustRate0");
        var qd = s.Find("JetDustRate0").Text;
        s.Expand("JetGrainExpander", "JetSizeQ");
        var q = s.Find("JetSizeQ").Text;
        output.WriteLine($"jet {j + 1}: Q_d at 1 AU = {qd}, q = {q}");
        Assert.True(double.TryParse(qd, NumberStyles.Float, CultureInfo.InvariantCulture, out var v) && v > 0,
          $"jet {j + 1}: Q_d '{qd}'");
        Assert.InRange(Ui.ParseNum(q), 2.0, 5.0);
      }
    });
  }

  /// <summary>
  /// The 1/ρ normalisation and the trajectory lines each change more than
  /// <see cref="OptionChangesTheFrame"/> of the frame, and switching them back restores it to
  /// within <see cref="OptionRestoresTheFrame"/> (paused, so only the option changes the picture).
  /// </summary>
  [SkippableFact]
  public void R8_RhoNormalisationAndTrajectories_ChangeTheFrame_AndSwitchBack()
  {
    var s = Shared();
    Guard("r8", () =>
    {
      ReviewSave.Pause(s);
      s.SelectTab("ModelTab");
      s.Expand("DustSection", "DustRhoNormToggle");
      var vp = s.Find("ViewportHost");
      Thread.Sleep(3000);
      var noise = FrameMetrics.Changed(FrameMetrics.Sample(vp), Wait(() => FrameMetrics.Sample(vp)));
      output.WriteLine($"paused, no change: {noise:P2} of the cells moved");
      foreach (var toggle in new[] { "DustRhoNormToggle", "DustTrajectoriesToggle" })
      {
        var before = FrameMetrics.Sample(vp);
        s.Find(toggle).RealClick();
        var on = Wait(() => FrameMetrics.Sample(vp));
        fixture.Screenshot($"r8-{toggle}.png");
        s.Find(toggle).RealClick();
        var back = Wait(() => FrameMetrics.Sample(vp));
        double changed = FrameMetrics.Changed(before, on), restored = FrameMetrics.Changed(before, back);
        output.WriteLine($"{toggle}: on {changed:P2}, back {restored:P2}");
        Assert.True(changed > OptionChangesTheFrame, $"{toggle} barely changed the frame ({changed:P2})");
        Assert.True(restored < OptionRestoresTheFrame, $"{toggle} switched back left {restored:P2} of the frame changed");
      }
    });
  }

  [SkippableFact]
  public void R9_SaveAsASingleAvk_ReopensWithItsJets()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    fixture.Close();
    var (artifacts, saveDir) = ReviewSave.Prepare("review-mail-avk");
    var avk = Path.Combine(artifacts, "review_67p.avk");

    using (var s = AppSession.Launch(Path.Combine(artifacts, "app-open.log"), diagnosticPanels: false))
    {
      try
      {
        Step("Save As a single .avk");
        ReviewSave.Open(s, Path.Combine(saveDir, "aethervk_index.json"));
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
        Step("reopen the .avk");
        ReviewSave.Open(s, avk);
        s.SelectTab("ModelTab");
        s.Expand("DustSection", "AddJetButton");
        Ui.WaitUntil(() => s.Items("JetListBox").Count == 4, TimeSpan.FromSeconds(60), "4 jet rows after reopen");
        Thread.Sleep(15000);
        Assert.DoesNotContain("Not in the simulation", s.App.PageSource);
        s.App.SaveScreenshot(Path.Combine(artifacts, "reopened-avk.png"));
      }
      catch
      {
        s.Desktop.SaveScreenshot(Path.Combine(artifacts, "failure-reopen.png"));
        throw;
      }
    }
  }

  /// <summary>
  /// A crash leaves a report: the app started with AETHERVK_E2E=1 and AETHERVK_TEST_CRASH=1
  /// throws an unhandled exception 3 s after its main window shows; within 10 s of the window an
  /// <c>aethervk-crash-&lt;stamp&gt;-&lt;pid&gt;.txt</c> is in the log folder
  /// (<c>%LOCALAPPDATA%\Aether-Vk\logs</c>) with the cause, and the run log names it. The next
  /// launch starts normally and writes no report of its own.
  /// </summary>
  [SkippableFact]
  public void R10_ACrash_LeavesAReportInTheLogFolder()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    fixture.Close();
    var artifacts = Path.Combine(AppSession.RepoRoot(), "TestResults", "e2e", "review-mail-crash");
    if (Directory.Exists(artifacts))
      Directory.Delete(artifacts, recursive: true);
    Directory.CreateDirectory(artifacts);
    var logs = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Aether-Vk", "logs");

    var exe = AppSession.AppPath();
    using var process = new Process
    {
      StartInfo = new ProcessStartInfo(exe) { WorkingDirectory = Path.GetDirectoryName(exe)!, UseShellExecute = false },
    };
    process.StartInfo.Environment["AETHERVK_E2E"] = "1";
    process.StartInfo.Environment["AETHERVK_TEST_CRASH"] = "1";
    Assert.True(process.Start(), $"could not start {exe}");
    try
    {
      Step("the main window");
      Ui.WaitUntil(() => process.HasExited || Win32.TopLevelWindows(process.Id).Any(w =>
          w.Title.StartsWith("AetherVk", StringComparison.Ordinal) && !w.Title.Contains("Starting", StringComparison.Ordinal)),
        TimeSpan.FromSeconds(90), "the main window");
      var shown = Stopwatch.StartNew();

      Step("the crash report");
      var pattern = $"aethervk-crash-*-{process.Id}.txt";
      var report = Ui.WaitFor(() => Directory.Exists(logs) ? Directory.GetFiles(logs, pattern).FirstOrDefault() : null,
        TimeSpan.FromSeconds(10), $"{pattern} in {logs}");
      output.WriteLine($"report after {shown.Elapsed.TotalSeconds:0.0} s: {report}");
      Thread.Sleep(1000); // written in one call; let the run log's flush land too
      var text = ReadShared(report);
      File.WriteAllText(Path.Combine(artifacts, Path.GetFileName(report)), text);
      Assert.Contains("Aether-Vk crash report: Unhandled exception", text);
      Assert.Contains("test crash requested through AETHERVK_TEST_CRASH", text);
      var runLog = RunLogLine().Match(text);
      Assert.True(runLog.Success, "the report does not name the run log");
      Assert.Contains("crash report written to " + report, ReadShared(runLog.Groups["path"].Value.Trim()));

      if (!process.WaitForExit(60_000))
        output.WriteLine("the crashed process had not exited after 60 s (Windows Error Reporting?)");
      else
        output.WriteLine($"the crashed process exited with 0x{process.ExitCode:x8}");
    }
    finally
    {
      if (!process.HasExited)
        process.Kill(entireProcessTree: true);
    }

    Step("the next launch");
    using var next = AppSession.Launch(Path.Combine(artifacts, "app-next.log"), diagnosticPanels: false);
    Thread.Sleep(10_000);
    next.Desktop.SaveScreenshot(Path.Combine(artifacts, "next-launch.png"));
    Assert.False(next.Process.HasExited, "the launch after the crash did not stay up");
    Assert.Empty(Directory.GetFiles(logs, $"aethervk-crash-*-{next.Process.Id}.txt"));
  }

  [GeneratedRegex(@"\d km( at the comet)?$")]
  private static partial Regex RulerInKm();

  [GeneratedRegex(@"^Run log: (?<path>.+)$", RegexOptions.Multiline)]
  private static partial Regex RunLogLine();

  /// <summary>Skips off Windows, else the shared session on the reviewer's save.</summary>
  private AppSession Shared()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    return fixture.Session(output);
  }

  /// <summary>Runs one item's steps, with a screenshot of the desktop when they fail.</summary>
  private void Guard(string item, Action steps)
  {
    Step(item);
    try
    {
      steps();
    }
    catch
    {
      fixture.Screenshot($"failure-{item}.png");
      throw;
    }
  }

  /// <summary>The frame 4 s on: the viewport redraws and the exposure settles.</summary>
  private static T Wait<T>(Func<T> sample)
  {
    Thread.Sleep(4000);
    return sample();
  }

  private static string ReadShared(string path)
  {
    using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
    return new StreamReader(stream).ReadToEnd();
  }

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
