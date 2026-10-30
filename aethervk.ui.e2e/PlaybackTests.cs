using System.Diagnostics;
using System.Globalization;
using Xunit.Abstractions;

namespace AetherVk.UI.E2E;

/// <summary>The playback tests' session: the reviewer's save at Low particle quality, its image hidden.</summary>
public sealed class PlaybackFixture : ReviewSaveFixture
{
  protected override string ArtifactsName => "playback";

  protected override bool LowQuality => true;
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
      Assert.StartsWith("2021-12-12", s.Find("TimelineStartEpochBox").Text);
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
