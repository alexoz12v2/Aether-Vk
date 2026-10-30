using System.Diagnostics;
using System.Globalization;
using Xunit.Abstractions;

namespace AetherVk.UI.E2E;

/// <summary>The playback tests' session: the reviewer's save at Low particle quality, its image hidden.</summary>
public sealed class PlaybackFixture : ReviewSaveFixture
{
  protected override string ArtifactsName => "playback";

  protected override bool LowQuality => true;

  protected override bool PerfLog => true;
}

/// <summary>
/// The Timeline tab's playback on the reviewer's 67P save (Low particle quality, the image
/// hidden): Play advances the clock at the chosen speed and changes the dust every frame, Pause
/// holds both, a speed change while playing changes the rate, Run to End and Reset reach the
/// committed ends, and a seek while paused redraws at the new epoch. Each test starts paused at
/// the committed start. Screenshots go to TestResults/e2e/playback/.
/// </summary>
[TestCaseOrderer(NaturalNameOrderer.TypeName, NaturalNameOrderer.AssemblyName)]
public sealed class PlaybackTests(PlaybackFixture fixture, ITestOutputHelper output) : IClassFixture<PlaybackFixture>
{
  /// <summary>Simulated seconds per wall-clock second at 1 h/s.</summary>
  private const double OneHourPerSecond = 3600.0;

  /// <summary>The measured rate may be this far off the chosen one (UIA reads take tens of ms).</summary>
  private const double RateTolerance = 0.25;

  private const int Frames = 5;
  private static readonly TimeSpan FrameGap = TimeSpan.FromMilliseconds(1500);

  /// <summary>Above this share of changed grid cells a frame pair counts as changed (as in ReviewerPlayTests).</summary>
  private const double ChangedFrame = 0.02;

  [SkippableFact]
  public void Play_AdvancesTheEpochAtTheChosenSpeed()
  {
    var s = Start();
    try
    {
      ReviewSave.Play(s);
      Thread.Sleep(2000);
      double rate = ReviewSave.MeasureRate(s, 5, TimeSpan.FromSeconds(2), output.WriteLine);
      ReviewSave.Pause(s);
      output.WriteLine($"rate {rate / OneHourPerSecond:0.000} h/s (chosen 1 h/s)");
      AssertRate(rate, OneHourPerSecond);
    }
    catch
    {
      fixture.Screenshot("failure-advances.png");
      throw;
    }
  }

  [SkippableFact]
  public void Play_ChangesTheDustEveryFrame()
  {
    var s = Start();
    try
    {
      var playing = PlayAndMeasure(s, Frames, "changes");
      ReviewSave.Pause(s);
      Assert.All(playing, c => Assert.True(c > ChangedFrame, $"a frame pair while playing barely changed ({c:P1})"));
    }
    catch
    {
      fixture.Screenshot("failure-changes.png");
      throw;
    }
  }

  [SkippableFact]
  public void Pause_HoldsTheEpochAndTheFrame()
  {
    var s = Start();
    try
    {
      var playing = PlayAndMeasure(s, 3, "pause");
      ReviewSave.Pause(s);
      Thread.Sleep(2000);
      var vp = s.Find("ViewportHost");
      var epoch = ReviewSave.Epoch(s);
      var a = FrameMetrics.Sample(vp);
      Thread.Sleep(3000);
      var b = FrameMetrics.Sample(vp);
      var later = ReviewSave.Epoch(s);
      double paused = FrameMetrics.Changed(a, b);
      output.WriteLine($"paused: epoch {epoch:u} -> {later:u}, changed {paused:P1}; playing average {playing.Average():P1}");
      Assert.Equal(epoch, later);
      Assert.True(paused < playing.Average() / 2, $"paused changed {paused:P1}, as much as playing ({playing.Average():P1})");
    }
    catch
    {
      fixture.Screenshot("failure-pause.png");
      throw;
    }
  }

  [SkippableFact]
  public void SpeedChange_WhilePlaying_ChangesTheRate()
  {
    var s = Start();
    try
    {
      ReviewSave.Play(s);
      Thread.Sleep(2000);
      double slow = ReviewSave.MeasureRate(s, 4, TimeSpan.FromSeconds(2), output.WriteLine);
      s.ChooseAt("SpeedPicker", ReviewSave.ThreeHoursPerSecondIndex);
      Assert.True(ReviewSave.IsPlaying(s), "choosing a speed stopped the playback");
      Thread.Sleep(1500);
      double fast = ReviewSave.MeasureRate(s, 4, TimeSpan.FromSeconds(2), output.WriteLine);
      ReviewSave.Pause(s);
      double ratio = fast / slow;
      output.WriteLine($"1 h/s measured {slow / OneHourPerSecond:0.000} h/s, 3 h/s measured {fast / OneHourPerSecond:0.000} h/s, ratio {ratio:0.000}");
      AssertRate(slow, OneHourPerSecond);
      Assert.InRange(ratio, 3.0 * (1 - RateTolerance), 3.0 * (1 + RateTolerance));
    }
    catch
    {
      fixture.Screenshot("failure-speed.png");
      throw;
    }
  }

  [SkippableFact]
  public void RunToEnd_ReachesTheCommittedEnd()
  {
    var s = Start();
    try
    {
      ReviewSave.Play(s);
      Thread.Sleep(2000);
      s.Click("RunToEndButton");
      var end = ReviewSave.EndEpoch(s);
      Ui.WaitUntil(() => ReviewSave.Near(ReviewSave.Epoch(s), end), TimeSpan.FromSeconds(120), $"the clock at the end {end:u}");
      Ui.WaitUntil(() => Ui.ReadRange(s.Find("TimelineProgress")) >= 99.9, TimeSpan.FromSeconds(10), "the progress bar at 100 %");
      fixture.Screenshot("run-to-end.png");
      output.WriteLine($"end {end:u}, clock {ReviewSave.Epoch(s):u}, progress {Ui.ReadRange(s.Find("TimelineProgress")).ToString(CultureInfo.InvariantCulture)}");
      Assert.False(ReviewSave.IsPlaying(s), "Run to End left it playing (the play button is not back to Play)");
    }
    catch
    {
      fixture.Screenshot("failure-run-to-end.png");
      throw;
    }
  }

  [SkippableFact]
  public void Reset_ReturnsToTheStart()
  {
    var s = Start();
    try
    {
      var start = ReviewSave.StartEpoch(s);
      Assert.True(ReviewSave.Shows(s.Find("TimelineStartEpochBox").Text, ReviewSave.CommittedStartUtc), s.Find("TimelineStartEpochBox").Text);
      ReviewSave.Play(s);
      Thread.Sleep(4000);
      ReviewSave.Pause(s);
      var played = ReviewSave.Epoch(s);
      Assert.True(played > start + TimeSpan.FromHours(1), $"the clock did not leave the start ({start:u} -> {played:u})");

      s.Click("ResetButton");
      Ui.WaitUntil(() => ReviewSave.Near(ReviewSave.Epoch(s), start), TimeSpan.FromSeconds(60), $"the clock back at {start:u}");
      output.WriteLine($"start {start:u}, played to {played:u}, reset to {ReviewSave.Epoch(s):u}");
      Assert.True(Ui.ReadRange(s.Find("TimelineProgress")) <= 0.1, "the progress bar is not back at 0");
    }
    catch
    {
      fixture.Screenshot("failure-reset.png");
      throw;
    }
  }

  [SkippableFact]
  public void SeekWhilePaused_RedrawsAtTheNewEpoch()
  {
    var s = Start();
    try
    {
      Thread.Sleep(3000);
      var vp = s.Find("ViewportHost");
      var before = FrameMetrics.Sample(vp);
      var target = ReviewSave.Epoch(s).Date.AddDays(3);
      var text = target.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);
      s.TypeInto(s.Find("CurrentEpochBox"), text, enter: false);
      s.Click("SeekToEpochButton");

      // the viewport's date label: "yyyy-MM-dd HH:mm (<scale>)", in the scale the box is typed in
      var label = target.ToString("yyyy-MM-dd HH:mm", CultureInfo.InvariantCulture);
      Ui.WaitUntil(() => ReviewSave.OverlayText(s, "ViewportEpochLabel")?.StartsWith(label, StringComparison.Ordinal) == true,
        TimeSpan.FromSeconds(60), $"the viewport label at {label}");
      Ui.WaitUntil(() => ReviewSave.Near(ReviewSave.Epoch(s), target), TimeSpan.FromSeconds(10), $"the epoch box at {text}");
      Thread.Sleep(4000);
      var after = FrameMetrics.Sample(vp);
      fixture.Screenshot("seek-paused.png");
      double changed = FrameMetrics.Changed(before, after);
      output.WriteLine($"seek to {text}: label \"{ReviewSave.OverlayText(s, "ViewportEpochLabel")}\", changed {changed:P1}");
      Assert.False(ReviewSave.IsPlaying(s), "the seek started the playback");
      Assert.True(changed > ChangedFrame, $"the frame after the seek barely differs from before ({changed:P1})");
    }
    catch
    {
      fixture.Screenshot("failure-seek.png");
      throw;
    }
  }

  /// <summary>
  /// After a pause the dust stays in preview for the 2 s hold, its circle filling beside "Render
  /// full"; then the full render draws ("Rendering the full image…") and settles. The phases come
  /// from the app's perf lines (once a second: a 2 s hold always shows), the circle from UIA.
  /// </summary>
  [SkippableFact]
  public void Pause_ShowsTheHold_ThenTheFullRender_ThenSettles()
  {
    var s = Start();
    try
    {
      s.Click("PlayPauseButton");
      Thread.Sleep(3000);
      var paused = DateTime.UtcNow;
      s.Click("PlayPauseButton");
      // best effort: a desktop-wide UIA search may take longer than the hold
      object? circle = null;
      try
      {
        circle = Ui.WaitFor(() => s.Desktop.FindElementsByAccessibilityId("DustHoldProgress").FirstOrDefault(e => e.Displayed),
          TimeSpan.FromSeconds(2), "the hold circle");
      }
      catch (TimeoutException)
      {
      }
      var phases = Phases(s, paused, TimeSpan.FromSeconds(60));
      output.WriteLine($"after the pause: {Describe(phases)}; hold circle {(circle is null ? "not seen" : "seen")}");
      int hold = phases.FindIndex(p => p.Phase == "hold");
      int rendering = phases.FindIndex(p => p.Phase == "rendering");
      Assert.True(hold >= 0, $"no hold after the pause: {Describe(phases)}");
      Assert.InRange(phases[hold].Hold!.Value, 0.0, 1.0);
      Assert.True(rendering > hold, $"no full render after the hold: {Describe(phases)}");
      // on the app's clock (a UIA click lands seconds late): the pause instant from the first
      // hold sample's progress, the full render at most a sample after the 2 s hold
      double pausedMs = phases[hold].TMs - phases[hold].Hold!.Value * 2000.0;
      double startMs = phases[rendering].TMs - pausedMs;
      Assert.True(startMs <= 3100,
        $"the full render started {startMs / 1000:0.0} s after the pause (hold 2 s): {Describe(phases)}");
      Assert.Equal("settled", phases[^1].Phase);
    }
    catch
    {
      fixture.Screenshot("failure-hold.png");
      throw;
    }
  }

  /// <summary>A typed epoch is one jump, not a scrub: the full render starts at once, no hold.</summary>
  [SkippableFact]
  public void EpochBoxSeek_StartsTheFullRender_WithoutAHold()
  {
    var s = Start();
    try
    {
      var target = ReviewSave.Epoch(s).Date.AddDays(2);
      var text = target.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);
      // from before the typing: the first lines are at the old epoch
      var since = DateTime.UtcNow;
      s.TypeInto(s.Find("CurrentEpochBox"), text, enter: false);
      s.Click("SeekToEpochButton");
      var phases = Phases(s, since, TimeSpan.FromSeconds(60));
      output.WriteLine($"after the seek: {Describe(phases)}");
      Assert.DoesNotContain(phases, p => p.Phase == "hold");
      int rendering = phases.FindIndex(p => p.Phase == "rendering");
      Assert.True(rendering >= 0, $"no full render after the seek: {Describe(phases)}");
      // on the app's clock (a UIA click lands seconds late): the seek is between the last line
      // at the old epoch and the first at the new one; the full render starts by the next sample
      int landed = phases.FindIndex(p => p.EpochS != phases[0].EpochS);
      Assert.True(landed > 0, $"the clock never moved: {Describe(phases)}");
      double startMs = phases[rendering].TMs - phases[landed - 1].TMs;
      Assert.True(startMs <= 2100,
        $"the full render started up to {startMs / 1000:0.0} s after the seek: {Describe(phases)}");
      Assert.Equal("settled", phases[^1].Phase);
    }
    catch
    {
      fixture.Screenshot("failure-seek-render.png");
      throw;
    }
  }

  /// <summary>The perf lines from <paramref name="since"/> until the dust has settled for 3 lines
  /// in a row after something else (or <paramref name="timeout"/>).</summary>
  private static List<PerfSample> Phases(AppSession s, DateTime since, TimeSpan timeout)
  {
    var limit = DateTime.UtcNow + timeout;
    while (true)
    {
      var lines = PerfLog.Since(s, since);
      bool busy = lines.Any(p => p.Phase is "hold" or "rendering");
      if (busy && lines.Count >= 3 && lines.TakeLast(3).All(p => p.Phase == "settled"))
        return lines;
      if (DateTime.UtcNow > limit)
        return lines;
      s.EnsureAlive("waiting for the dust to settle");
      Thread.Sleep(250);
    }
  }

  private static string Describe(List<PerfSample> phases) =>
    string.Join(" ", phases.Select(p => (p.Phase == "hold" ? $"hold({p.Hold:0.00})" : p.Phase ?? "null") + $"@{p.TMs / 1000:0.0}"));

  /// <summary>Skips off Windows, else the shared session paused at the committed start, 1 h/s chosen.</summary>
  private AppSession Start()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var s = fixture.Session(output);
    ReviewSave.ResetToStart(s);
    s.ChooseAt("SpeedPicker", ReviewSave.OneHourPerSecondIndex);
    return s;
  }

  /// <summary>Plays and returns the changed share of <paramref name="pairs"/> consecutive frame pairs.</summary>
  private List<double> PlayAndMeasure(AppSession s, int pairs, string name)
  {
    var vp = s.Find("ViewportHost");
    ReviewSave.Play(s);
    Thread.Sleep(2000);
    var playing = new List<double>();
    var previous = FrameMetrics.Sample(vp);
    for (int i = 1; i <= pairs; i++)
    {
      Thread.Sleep(FrameGap);
      var frame = FrameMetrics.Sample(vp);
      playing.Add(FrameMetrics.Changed(previous, frame));
      previous = frame;
    }
    fixture.Screenshot($"{name}-playing.png");
    output.WriteLine($"changed while playing: {string.Join(", ", playing.Select(c => c.ToString("P1", CultureInfo.InvariantCulture)))}");
    return playing;
  }

  private static void AssertRate(double measured, double chosen) =>
    Assert.True(Math.Abs(measured / chosen - 1) <= RateTolerance,
      $"the clock ran at {measured / 3600.0:0.000} h/s for a chosen {chosen / 3600.0:0.000} h/s (±{RateTolerance:P0})");
}

/// <summary>
/// The dust history (the diagnostic panel's DUST HISTORY line, so this one runs with the
/// diagnostic panels on, in a session of its own): once playing, no age tier is still
/// "building" within 20 s.
/// </summary>
public sealed class DustHistoryPlaybackTests(ITestOutputHelper output)
{
  private static readonly TimeSpan BuildingLimit = TimeSpan.FromSeconds(20);

  [SkippableFact]
  public void Play_DustHistoryLeavesBuilding()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var (artifacts, saveDir) = ReviewSave.Prepare("playback-dust-history");
    using var preferences = new PreferencesBackup();
    using var s = AppSession.Launch(Path.Combine(artifacts, "app.log"), diagnosticPanels: true);
    try
    {
      ReviewSave.Open(s, Path.Combine(saveDir, "aethervk_index.json"));
      ReviewSave.SetLowQuality(s);
      ReviewSave.ResetToStart(s);
      s.ChooseAt("SpeedPicker", ReviewSave.OneHourPerSecondIndex);

      ReviewSave.Play(s);
      var clock = Stopwatch.StartNew();
      string last = string.Empty;
      while (clock.Elapsed < BuildingLimit)
      {
        last = ReviewSave.OverlayText(s, "DustHistoryText") ?? string.Empty;
        if (last.Length > 0 && !last.Contains("building", StringComparison.Ordinal))
          break;
        Thread.Sleep(500);
      }
      var elapsed = clock.Elapsed;
      ReviewSave.Pause(s);
      output.WriteLine($"after {elapsed.TotalSeconds:0.0} s of play the dust history reads:\n{last}");
      s.Desktop.SaveScreenshot(Path.Combine(artifacts, "dust-history.png"));
      Assert.False(string.IsNullOrEmpty(last), "no DUST HISTORY line while playing");
      Assert.DoesNotContain("building", last, StringComparison.Ordinal);
    }
    catch
    {
      s.Desktop.SaveScreenshot(Path.Combine(artifacts, "failure.png"));
      throw;
    }
  }
}
