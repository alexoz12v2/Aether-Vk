using System.Diagnostics;
using System.Globalization;
using System.Text;
using Xunit.Abstractions;

namespace AetherVk.UI.E2E;

/// <summary>
/// Playback of the reviewer's 67P save, measured: 10 s at 10 s/s (the slowest speed: the dust's
/// slow evolution must still show on screen) and 10 s at 1 d/s, each after a 2 s warm-up, at Low
/// particle quality and the Balanced playback preview, image hidden, from two days past the
/// committed start. Each second it captures the screen around the comet and reads the app's
/// <c>[perf]</c> line (<c>AETHERVK_PERF_LOG=1</c>): the picture must move, the dust frame rate and
/// frame time must hold limits scaled by the machine's capability score (calibrated paused
/// against perf_reference.json, a GTX 1070 = 1), the clock must run at the chosen speed and the
/// comet must not jump. With <c>AETHERVK_E2E_PERF_RECORD=1</c> it records a suggested reference
/// and asserts nothing. Artifacts in TestResults/e2e/playback-perf/.
/// </summary>
public sealed class PlaybackPerfTests(ITestOutputHelper output)
{
  /// <summary>Sim seconds per wall second at the two speeds.</summary>
  private const double TenSecondsPerSecond = 10.0, OneDayPerSecond = 86400.0;

  /// <summary>The Balanced preview's frame budget (ms) and its index in PlayPreviewPicker (Off, Smooth, Balanced, Quality).</summary>
  private const double BudgetMs = 33.0;
  private const int BalancedIndex = 2;

  private static readonly TimeSpan WarmUp = TimeSpan.FromSeconds(2);
  private const int Seconds = 10;

  /// <summary>Calibration seeks (paused): each one's full render is a fixed workload, timed once.</summary>
  private static readonly double[] CalibrationDays = [2.0, 2.25, 2.5];

  /// <summary>A perf line times a full dust render when its splat pass ran (a settled paused frame only composites).</summary>
  private const double RenderedSplatMs = 1.0;

  /// <summary>The paused image counts as steady below this change in 1 s (mean luminance).</summary>
  private const double SteadyMad = 0.002;

  /// <summary>How late a 1 s capture tick may start (the loop's own cost).</summary>
  private const double MaxTickLateS = 0.5;

  /// <summary>The last frame that drew, drawn with every cluster (not strided).</summary>
  private static bool IsSettledFullFrame(PerfSample p) =>
    p.DrawnFrame is not null && p.DrawnStride == 1 && p.DrawnSplatMs > RenderedSplatMs;

  /// <summary>J2000 TDB, the origin of the perf line's <c>epoch_s</c>.</summary>
  private static readonly DateTime J2000 = new(2000, 1, 1, 12, 0, 0, DateTimeKind.Utc);

  /// <summary>TDB − UTC in 2021 (32 leap seconds + 32.184 s).</summary>
  private const double TdbMinusUtc2021 = 69.184;

  // ── limits ──

  /// <summary>The dust frame rate a score of 1 must reach (a 60 Hz display), its floor and cap.</summary>
  private const double FpsPerScore = 60.0, FpsFloor = 20.0, FpsCap = 55.0;

  /// <summary>The clock may run this far off the chosen speed.</summary>
  private const double RateTolerance = 0.10;

  /// <summary>At 10 s/s: the first and last frame differ by at least this mean luminance (or 3 × the paused noise).</summary>
  private const double SlowDriftMin = 0.002;

  /// <summary>At 10 s/s: the median second-to-second change is at least this (or 2 × the noise): it keeps moving.</summary>
  private const double SlowStepMin = 0.001;

  /// <summary>No second-to-second change above this mean luminance (a flash, not motion).</summary>
  private const double FlickerMax = 0.05;

  /// <summary>At 1 d/s: every second-to-second pair changes at least this share of cells.</summary>
  private const double FastChangedMin = 0.02;

  /// <summary>The comet may step this much more than its median step, and never more than <see cref="CometStepMaxPx"/>.</summary>
  private const double CometStepFactor = 3.0, CometStepSlackPx = 4.0, CometStepMaxPx = 40.0;

  /// <summary>The "physics behind" line of the logic thread (rate limited: any one counts).</summary>
  private const string BacklogLine = "Physics step backlog dropped";

  [SkippableFact]
  public void Playback_MovesAndHoldsTheFrameRate()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var (artifacts, saveDir) = ReviewSave.Prepare("playback-perf");
    using var preferences = new PreferencesBackup();
    var reference = PerfReference.Load();
    // the diagnostic panel carries the comet's screen position (the observer HUD)
    using var s = AppSession.Launch(Path.Combine(artifacts, "app.log"), diagnosticPanels: true, perfLog: true);
    try
    {
      ReviewSave.Open(s, Path.Combine(saveDir, "aethervk_index.json"));
      ReviewSave.SetQuality(s, 2, "Low");
      s.Expand("ParticleRenderingSection", "PlayPreviewPicker");
      s.ChooseAt("PlayPreviewPicker", BalancedIndex);
      ReviewSave.HideTheImage(s);
      ReviewSave.ResetToStart(s);

      // ── calibration, paused: the settled full render after a seek, at three epochs. A settled
      // paused frame reuses its image (< 1 ms, says nothing of the GPU), so the cost is the last
      // frame that drew: the refined image, every cluster (stride 1), preview over. Settled = the
      // same last drawn frame on two perf lines in a row ──
      var fullFrames = new List<double>();
      foreach (var days in CalibrationDays)
      {
        var since = SeekTo(s, ReviewSave.CommittedStartUtc.AddDays(days));
        var settled = Ui.WaitFor(() =>
        {
          var lines = PerfLog.Since(s, since);
          for (int i = lines.Count - 1; i > 0; i--)
          {
            var (a, b) = (lines[i - 1], lines[i]);
            if (a.Preview == false && b.Preview == false && IsSettledFullFrame(a) && b.DrawnFrame == a.DrawnFrame)
              return b;
          }
          return null;
        }, TimeSpan.FromSeconds(60), $"the settled full render after the seek to +{days} d");
        fullFrames.Add(settled.DrawnTotalMs!.Value);
      }
      output.WriteLine("full render after a seek: " + string.Join(", ", fullFrames.Select(x => $"{x:0.0} ms")));
      double calibMs = Median(fullFrames);
      var gpu = PerfLog.GpuName(s);
      var capability = Capability.Measure(reference, gpu, calibMs);
      output.WriteLine(capability.ToString());
      File.WriteAllText(Path.Combine(artifacts, "calibration.json"),
        (reference with { Gpu = gpu, CalibTotalMs = Math.Round(calibMs, 3) }).ToJson());

      var region = Region(s);
      // the paused flicker, once the image is steady: after the last drawn frame the composite's
      // exposure still adapts for a few seconds (not flicker, and it would inflate the floor)
      var noise = double.NaN;
      var steadyBy = DateTime.UtcNow + TimeSpan.FromSeconds(20);
      var grid = Grid(region);
      do
      {
        var next = Pause(1000, () => Grid(region));
        noise = MeanAbsDiff(grid, next);
        grid = next;
      } while (noise > SteadyMad && DateTime.UtcNow < steadyBy);
      output.WriteLine($"paused noise {noise:0.00000} (mean luminance), region {region}");
      Assert.True(noise <= SteadyMad, $"the paused image never held still: {noise:0.00000} mean luminance change in 1 s after 20 s, limit {SteadyMad}");

      // ── 10 s/s, then 1 d/s ──
      var slow = Run(s, artifacts, "10s-per-s", ReviewSave.TenSecondsPerSecondIndex, region);
      // 1 d/s covers ~15 days in the phase: from the committed start, so it ends before the
      // committed end (16 days later) and never measures a clock stopped at the end
      SeekTo(s, ReviewSave.CommittedStartUtc.AddHours(1));
      Thread.Sleep(3000);
      var fast = Run(s, artifacts, "1d-per-s", ReviewSave.OneDayPerSecondIndex, region);
      WriteCsv(Path.Combine(artifacts, "perf.csv"), [("10s-per-s", slow), ("1d-per-s", fast)]);

      if (PerfReference.Recording)
      {
        var suggested = reference with { Gpu = gpu, CalibTotalMs = Math.Round(calibMs, 3) };
        File.WriteAllText(Path.Combine(artifacts, "perf_reference.suggested.json"), suggested.ToJson());
        output.WriteLine($"recording: suggested reference\n{suggested.ToJson()}");
        return;
      }

      var failures = new List<string>();
      CheckPerf(slow, TenSecondsPerSecond, capability, failures);
      CheckPerf(fast, OneDayPerSecond, capability, failures);
      CheckSlowMovement(slow, noise, failures);
      CheckFastMovement(fast, failures);
      CheckComet(slow, failures);
      CheckComet(fast, failures);
      Assert.True(failures.Count == 0, $"{capability}\n" + string.Join("\n", failures));
    }
    catch
    {
      try { s.Desktop.SaveScreenshot(Path.Combine(artifacts, "failure.png")); } catch (Exception) { }
      throw;
    }
  }

  // ── a phase ──

  /// <summary>The frames, comet positions and perf lines of one speed's measured seconds.</summary>
  private sealed record Phase(
    string Name, double Speed, List<double[]> Frames, List<(double X, double Y)> Comet, List<PerfSample> Perf, int BacklogLines);

  private Phase Run(AppSession s, string artifacts, string name, int speedIndex, Rect region)
  {
    s.SelectTab("TimelineTab");
    s.ChooseAt("SpeedPicker", speedIndex);
    double speed = speedIndex == ReviewSave.TenSecondsPerSecondIndex ? TenSecondsPerSecond : OneDayPerSecond;
    // playing is read from the perf line's clock, not the UI: ReviewSave.Play looks for the Reset
    // button's absence, a whole-tree UIA search that takes seconds under load (at 1 d/s, days of
    // the range gone before the measurement starts)
    var clicked = DateTime.UtcNow;
    s.Click("PlayPauseButton");
    Ui.WaitUntil(() => PerfLog.Since(s, clicked).Select(p => p.EpochS).OfType<double>().Distinct().Count() >= 2,
      TimeSpan.FromSeconds(10), $"[{name}] the clock to run");
    var warm = WarmUp - (DateTime.UtcNow - clicked);
    if (warm > TimeSpan.Zero)
      Thread.Sleep(warm);

    // looked up once: a desktop-wide UIA search and a desktop screenshot take seconds each, a tick
    // has one second (the HUD text read and two BitBlts stay well inside it)
    var vp = s.Find("ViewportHost");
    var (vpAt, vpSize) = (vp.Location, vp.Size);
    var hud = s.Desktop.FindElementsByAccessibilityId("EarthObserverStatus").FirstOrDefault();
    var since = DateTime.UtcNow;
    var frames = new List<double[]>();
    var comet = new List<(double, double)>();
    var late = new List<double>();
    var clock = Stopwatch.StartNew();
    for (int i = 0; i <= Seconds; i++)
    {
      var due = TimeSpan.FromSeconds(i) - clock.Elapsed;
      if (due > TimeSpan.Zero)
        Thread.Sleep(due);
      late.Add(-due.TotalSeconds);
      frames.Add(Grid(region));
      comet.Add(FitsComparisonSteps.CometScreenPx(hud?.Text, vpAt, vpSize));
      if (i % 2 == 0)
        Win32.SavePng(Path.Combine(artifacts, $"{name}-{i:00}.png"), vpAt.X, vpAt.Y, vpSize.Width, vpSize.Height);
    }
    var until = DateTime.UtcNow;
    ReviewSave.Pause(s);
    // a tick that started late stretches the phase: the rate and movement checks assume 1 s ticks
    if (late.Max() > MaxTickLateS)
      throw new InvalidOperationException(
        $"[{name}] the capture loop fell behind ({late.Max():0.00} s late at worst, limit {MaxTickLateS} s): the measurement is not {Seconds} s");

    var perf = PerfLog.Since(s, since).Where(p => p.At <= until).ToList();
    int backlog = s.Lines.Count(l => l.At >= since - WarmUp && l.At <= until && l.Line.Contains(BacklogLine, StringComparison.Ordinal));
    var phase = new Phase(name, speed, frames, comet, perf, backlog);
    output.WriteLine(Summary(phase));
    return phase;
  }

  // ── checks ──

  private static void CheckPerf(Phase p, double speed, Capability c, List<string> failures)
  {
    string at = $"[{p.Name}]";
    if (p.Perf.Count < Seconds / 2)
    {
      failures.Add($"{at} only {p.Perf.Count} [perf] lines in {Seconds} s");
      return;
    }
    double fps = Median(Fps(p.Perf));
    double fpsMin = Math.Max(FpsFloor, Math.Min(FpsCap, FpsPerScore * c.Score));
    if (!(fps >= fpsMin))
      failures.Add($"{at} dust {fps:0.0} fps (median), limit ≥ {fpsMin:0.0} (score {c.Score:0.00})");

    var totals = p.Perf.Select(x => x.TotalMs).OfType<double>().ToList();
    double median = Median(totals), p95 = Percentile(totals, 0.95);
    double medianMax = BudgetMs / c.Score, p95Max = 2 * BudgetMs / c.Score;
    if (!(median <= medianMax))
      failures.Add($"{at} dust frame {median:0.0} ms (median), limit ≤ {medianMax:0.0} (score {c.Score:0.00})");
    if (!(p95 <= p95Max))
      failures.Add($"{at} dust frame {p95:0.0} ms (p95), limit ≤ {p95Max:0.0} (score {c.Score:0.00})");

    int preview = p.Perf.Count(x => x.Preview == true);
    if (preview < p.Perf.Count * 0.8)
      failures.Add($"{at} the dust was in preview in {preview} of {p.Perf.Count} perf lines while playing");

    double rate = Rate(p.Perf);
    if (!(Math.Abs(rate / speed - 1) <= RateTolerance))
      failures.Add($"{at} the clock ran at {rate:0.###} s/s for a chosen {speed:0.###} s/s (±{RateTolerance:P0})");
    if (p.BacklogLines > 0)
      failures.Add($"{at} {p.BacklogLines} \"{BacklogLine}\" lines");
  }

  private static void CheckSlowMovement(Phase p, double noise, List<string> failures)
  {
    var steps = Steps(p.Frames, MeanAbsDiff);
    double drift = MeanAbsDiff(p.Frames[0], p.Frames[^1]);
    double driftMin = Math.Max(3 * noise, SlowDriftMin), stepMin = Math.Max(2 * noise, SlowStepMin);
    if (!(drift >= driftMin))
      failures.Add($"[{p.Name}] first to last frame {drift:0.00000}, limit ≥ {driftMin:0.00000}: the dust does not evolve");
    if (!(Median(steps) >= stepMin))
      failures.Add($"[{p.Name}] median second-to-second change {Median(steps):0.00000}, limit ≥ {stepMin:0.00000}: no sense of movement");
    if (steps.Max() > FlickerMax)
      failures.Add($"[{p.Name}] a second-to-second change of {steps.Max():0.0000} > {FlickerMax}: flicker");
  }

  private static void CheckFastMovement(Phase p, List<string> failures)
  {
    var changed = Steps(p.Frames, Changed);
    for (int i = 0; i < changed.Count; i++)
      if (!(changed[i] >= FastChangedMin))
        failures.Add($"[{p.Name}] second {i}→{i + 1} changed {changed[i]:P1} of the cells, limit ≥ {FastChangedMin:P0}");
  }

  private static void CheckComet(Phase p, List<string> failures)
  {
    var steps = Steps(p.Comet, (a, b) => Math.Sqrt((a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y)));
    double limit = Math.Min(CometStepMaxPx, CometStepFactor * Median(steps) + CometStepSlackPx);
    for (int i = 0; i < steps.Count; i++)
      if (steps[i] > limit)
        failures.Add($"[{p.Name}] the comet jumped {steps[i]:0.0} px at second {i}→{i + 1} ({p.Comet[i]} → {p.Comet[i + 1]}), limit {limit:0.0}");
  }

  // ── screen ──

  private readonly record struct Rect(int Left, int Top, int Width, int Height);

  /// <summary>A square of half the viewport's short side around the comet, inside the viewport.</summary>
  private static Rect Region(AppSession s)
  {
    var vp = s.Find("ViewportHost");
    var (cx, cy) = FitsComparisonSteps.CometScreenPx(s);
    int side = Math.Min(vp.Size.Width, vp.Size.Height) / 2;
    int left = Math.Clamp((int)cx - side / 2, vp.Location.X, vp.Location.X + vp.Size.Width - side);
    int top = Math.Clamp((int)cy - side / 2, vp.Location.Y, vp.Location.Y + vp.Size.Height - side);
    return new Rect(left, top, side, side);
  }

  /// <summary>The region's luminance (0..1) averaged on a <see cref="FrameMetrics.GridSize"/>² grid.</summary>
  private static double[] Grid(Rect r)
  {
    var luma = Win32.CaptureLuminance(r.Left, r.Top, r.Width, r.Height);
    int n = FrameMetrics.GridSize;
    var sum = new double[n * n];
    var count = new int[n * n];
    for (int y = 0; y < r.Height; y++)
      for (int x = 0; x < r.Width; x++)
      {
        int cell = y * n / r.Height * n + x * n / r.Width;
        sum[cell] += luma[y * r.Width + x];
        count[cell]++;
      }
    for (int i = 0; i < sum.Length; i++)
      sum[i] /= Math.Max(1, count[i]);
    return sum;
  }

  private static double MeanAbsDiff(double[] a, double[] b) => a.Zip(b, (x, y) => Math.Abs(x - y)).Average();

  private static double Changed(double[] a, double[] b) => FrameMetrics.Changed(a, b);

  // ── numbers ──

  private static List<double> Steps<T>(List<T> xs, Func<T, T, double> d) =>
    Enumerable.Range(1, xs.Count - 1).Select(i => d(xs[i - 1], xs[i])).ToList();

  /// <summary>Dust frames per second between consecutive perf lines (the app's own clock).</summary>
  private static List<double> Fps(List<PerfSample> perf) =>
    Enumerable.Range(1, Math.Max(0, perf.Count - 1))
      .Where(i => perf[i].DustFrame is not null && perf[i - 1].DustFrame is not null && perf[i].TMs > perf[i - 1].TMs)
      .Select(i => (double)(perf[i].DustFrame!.Value - perf[i - 1].DustFrame!.Value) / ((perf[i].TMs - perf[i - 1].TMs) * 1e-3))
      .ToList();

  /// <summary>Sim seconds per wall second from the first to the last perf line with a clock.</summary>
  private static double Rate(List<PerfSample> perf)
  {
    var withEpoch = perf.Where(p => p.EpochS is not null).ToList();
    if (withEpoch.Count < 2)
      return double.NaN;
    var (a, b) = (withEpoch[0], withEpoch[^1]);
    return (b.EpochS!.Value - a.EpochS!.Value) / ((b.TMs - a.TMs) * 1e-3);
  }

  private static double Median(IEnumerable<double> xs) => Percentile(xs.ToList(), 0.5);

  private static double Percentile(List<double> xs, double q)
  {
    if (xs.Count == 0)
      return double.NaN;
    var sorted = xs.OrderBy(x => x).ToList();
    double at = q * (sorted.Count - 1);
    int lo = (int)Math.Floor(at), hi = (int)Math.Ceiling(at);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (at - lo);
  }

  // ── steps ──

  private static T Pause<T>(int ms, Func<T> then)
  {
    Thread.Sleep(ms);
    return then();
  }

  /// <summary>Paused, types <paramref name="utc"/> into the epoch box and seeks (as PlaybackTests' seek).</summary>
  /// <summary>Seeks to <paramref name="utc"/> and waits until the simulation clock is there (the
  /// perf line's TDB epoch, not the box just typed into); returns the time of the click.</summary>
  private static DateTime SeekTo(AppSession s, DateTime utc)
  {
    s.SelectTab("TimelineTab");
    var text = utc.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);
    s.TypeInto(s.Find("CurrentEpochBox"), text, enter: false);
    var clicked = DateTime.UtcNow;
    s.Click("SeekToEpochButton");
    double want = (utc - J2000).TotalSeconds + TdbMinusUtc2021;
    Ui.WaitUntil(() => PerfLog.Since(s, clicked).Any(p => p.EpochS is { } e && Math.Abs(e - want) < 2),
      TimeSpan.FromSeconds(60), $"the simulation clock at {text} UTC");
    return clicked;
  }

  private static string Summary(Phase p)
  {
    var c = CultureInfo.InvariantCulture;
    var sb = new StringBuilder($"[{p.Name}] {p.Perf.Count} perf lines; ");
    sb.Append(c, $"fps median {Median(Fps(p.Perf)):0.0}; ");
    var totals = p.Perf.Select(x => x.TotalMs).OfType<double>().ToList();
    sb.Append(c, $"total ms median {Median(totals):0.0} p95 {Percentile(totals, 0.95):0.0}; ");
    sb.Append(c, $"rate {Rate(p.Perf):0.###} s/s (chosen {p.Speed}); backlog lines {p.BacklogLines}\n");
    sb.Append("  change/s (MAD): ").AppendJoin(", ", Steps(p.Frames, MeanAbsDiff).Select(x => x.ToString("0.00000", c))).Append('\n');
    sb.Append("  change/s (cells): ").AppendJoin(", ", Steps(p.Frames, Changed).Select(x => x.ToString("P1", c))).Append('\n');
    sb.Append(c, $"  first→last MAD {MeanAbsDiff(p.Frames[0], p.Frames[^1]):0.00000}\n");
    sb.Append("  comet px: ").AppendJoin(" ", p.Comet.Select(x => $"({x.X:0},{x.Y:0})"));
    return sb.ToString();
  }

  private static void WriteCsv(string path, IEnumerable<(string Phase, Phase P)> phases)
  {
    var sb = new StringBuilder("phase," + PerfSample.CsvHeader + "\n");
    foreach (var (name, p) in phases)
      foreach (var x in p.Perf)
        sb.Append(name).Append(',').Append(x.Csv()).Append('\n');
    File.WriteAllText(path, sb.ToString());
  }
}
