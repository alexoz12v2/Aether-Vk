using System.Globalization;
using System.IO.Compression;
using AetherVk.Logic.Models;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using Xunit;
using Xunit.Abstractions;
using static AetherVk.X11Tests.X11Host;

namespace AetherVk.X11Tests;

/// <summary>
/// The reviewer's slow-rotator scenario as a save: a comet about 13.5 AU from the Sun, a 60 km
/// nucleus radius (C/2014 UN271's), a 120 h nucleus rotation, 73 rotations (365 d) of model range
/// ending at 2026-11-01 UTC, jets, the dust simulated to the end, saved as one <c>.avk</c> and
/// reopened. The comet is not named in the
/// question: the test picks it through the app's own Horizons/SBDB path (the Comet tab's search
/// and <see cref="HorizonJplService.FetchSmallBodyDataAsync"/>) as the candidate whose
/// heliocentric distance at the observation date is closest to 13.5 AU, and fails with every
/// candidate's distance when none is within <see cref="DistanceToleranceAu"/>.
///
/// <para>Needs the network (SBDB, Horizons). With <c>AETHERVK_WRITE_FIXTURE=1</c> the archive is
/// written to <c>test_assets/saves/review_slow_rotator_2026-11.avk</c>; otherwise to a temp
/// folder.</para>
/// </summary>
[Collection("X11")]
public sealed class SlowRotatorSaveX11Tests : IDisposable
{
  private readonly ITestOutputHelper _output;

  public SlowRotatorSaveX11Tests(ITestOutputHelper output)
  {
    _output = output;
    Skip.If(SkipReason is not null, SkipReason);
    EnsureStarted();
  }

  public void Dispose()
  {
  }

  // ── The scenario ───────────────────────────────────────────────────────────

  /// <summary>The observation date (the end of the range), UTC, in the Observation box format.</summary>
  public const string ObservationUtc = "2026-11-01T00:00:00";

  public const double RotationPeriodHours = 120;
  public const int Rotations = 73;
  /// <summary>The nucleus radius (km) of C/2014 UN271, set through the Comet tab's slider binding;
  /// the engine computes the saved state with it (compatibility JSON <c>nucleus_radius_km</c>).</summary>
  public const float NucleusRadiusKm = 60f;
  public const double TargetDistanceAu = 13.5;
  public const double DistanceToleranceAu = 1.5;

  /// <summary>
  /// Comets known to be far out and active, as the SBDB names them (the year-letter part matches
  /// both the <c>pdes</c> and the full name). The test does not assume which one is at 13.5 AU.
  /// </summary>
  private static readonly string[] Candidates =
  [
    "2014 UN271", // Bernardinelli–Bernstein, inbound to a 2031 perihelion at 11 AU
    "2017 K2",    // PANSTARRS, outbound since its 2022 perihelion
    "2010 U3",    // Boattini, outbound since its 2019 perihelion at 8.4 AU
    "2019 L3",    // ATLAS, outbound since 2022
    "2020 F2",    // ATLAS, perihelion 2022 at 8.8 AU
  ];

  /// Where <c>AETHERVK_WRITE_FIXTURE=1</c> writes the archive: the checkout's test_assets/ (repack
  /// the saves group with scripts/pack_test_assets.sh afterwards); readers go through ASSETS.tsv.
  public static string FixturePath() =>
    Path.Combine(AppRig.RepoRoot(), "test_assets", "saves", "review_slow_rotator_2026-11.avk");

  private static bool WriteFixture => Environment.GetEnvironmentVariable("AETHERVK_WRITE_FIXTURE") == "1";

  // ── Heliocentric distance from the SBDB elements ───────────────────────────

  /// <summary>Gaussian gravitational constant, AU^1.5 / day.</summary>
  private const double GaussK = 0.01720209895;

  private static double JulianDay(DateTimeOffset t) =>
    2451545.0 + (t - new DateTimeOffset(2000, 1, 1, 12, 0, 0, TimeSpan.Zero)).TotalDays;

  /// <summary>
  /// r_h (AU) at <paramref name="t"/> from the perihelion distance, eccentricity and perihelion
  /// time (two-body motion about the Sun; the conic from the sign of e − 1). NaN without a
  /// perihelion time.
  /// </summary>
  internal static double HeliocentricDistanceAu(double q, double e, double tpJd, DateTimeOffset t)
  {
    if (double.IsNaN(tpJd) || !(q > 0) || e < 0)
      return double.NaN;
    double dt = JulianDay(t) - tpJd;
    if (Math.Abs(e - 1) < 1e-9)
    {
      // Barker: D + D³/3 = k Δt / √(2 q³)
      double s = 1.5 * GaussK * dt / Math.Sqrt(2 * q * q * q);
      double root = Math.Sqrt(s * s + 1);
      double d = Math.Cbrt(s + root) + Math.Cbrt(s - root);
      return q * (1 + d * d);
    }
    if (e < 1)
    {
      double a = q / (1 - e);
      double m = GaussK / Math.Pow(a, 1.5) * dt;
      m = Math.IEEERemainder(m, 2 * Math.PI);
      double ea = e > 0.8 ? Math.PI * Math.Sign(m == 0 ? 1 : m) : m;
      for (int i = 0; i < 100; i++)
      {
        double f = ea - e * Math.Sin(ea) - m;
        double step = f / (1 - e * Math.Cos(ea));
        ea -= step;
        if (Math.Abs(step) < 1e-14) break;
      }
      return a * (1 - e * Math.Cos(ea));
    }
    else
    {
      double a = q / (e - 1);
      double m = GaussK / Math.Pow(a, 1.5) * dt;
      double h = Math.Asinh(m / e);
      for (int i = 0; i < 100; i++)
      {
        double f = e * Math.Sinh(h) - h - m;
        double step = f / (e * Math.Cosh(h) - 1);
        h -= step;
        if (Math.Abs(step) < 1e-14) break;
      }
      return a * (e * Math.Cosh(h) - 1);
    }
  }

  private sealed record Candidate(string Key, CometSearchResult? Listed, SmallBodyDataComponent? Sbdb, double DistanceAu, string Note);

  private void Step(string s) => _output.WriteLine($"── {s}");

  private static bool Clean(string title) => !title.TrimEnd().EndsWith('*');

  /// <summary>A box (UTC scale, whole seconds) names <paramref name="expected"/> to the second.</summary>
  private static void AssertBox(DateTimeOffset expected, string text)
  {
    Assert.True(TimeUtils.TryParseIso8601(text, out var t), $"not a date: '{text}'");
    Assert.True(Math.Abs((t - expected).TotalSeconds) <= 1, $"'{text}' is not {expected:O}");
  }

  /// <summary>The engine's <c>nucleus_radius_km</c> from a compatibility JSON, NaN when absent.</summary>
  private static double CompatRadiusKm(string? json)
  {
    if (string.IsNullOrEmpty(json)) return double.NaN;
    using var doc = System.Text.Json.JsonDocument.Parse(json);
    return doc.RootElement.TryGetProperty("nucleus_radius_km", out var r) ? r.GetDouble() : double.NaN;
  }

  private static void AwaitUi(Task task, int timeoutMs, string what)
  {
    Assert.True(task.Wait(TimeSpan.FromMilliseconds(timeoutMs)), $"{what}: not done after {timeoutMs / 1000} s");
    Assert.True(task.IsCompletedSuccessfully, $"{what}: {task.Exception?.GetBaseException().Message}");
  }

  /// A comet at ≈ 13.5 AU is picked from the Comet tab's list by its SBDB elements, its rotation
  /// period set to 120 h, the Observation block set to 2026-11-01 and 73 rotations (a 365 d
  /// range), the comet committed with it, the nucleus radius set to 60 km, two jets added in the
  /// Model tab, the clock run to the observation date, File › Save As .avk; then File › New and
  /// the archive reopened: the same range, period, rotation count, observation date, radius and
  /// jets, with the saved scene (computed with the 60 km nucleus) restored.
  [SkippableFact]
  public void SlowRotatorYearLongRange_SavesAndReopensAsAvk()
  {
    using var rig = new AppRig();
    var avk = WriteFixture ? FixturePath() : Path.Combine(rig.SavesDir, "review_slow_rotator_2026-11.avk");
    var observation = DateTimeOffset.ParseExact(ObservationUtc, RangeEntryViewModel.UtcBoxFormat, CultureInfo.InvariantCulture,
      DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal);
    var expectedStart = observation - TimeSpan.FromHours(Rotations * RotationPeriodHours);

    var comet = rig.Tab<CometTabViewModel>();
    var model = rig.Tab<ModelTabViewModel>();
    var jpl = rig.Host.Services.GetRequiredService<HorizonJplService>();
    var runtime = rig.Host.Services.GetRequiredService<INativeRuntimeService>();

    // ── 1. the comet: the Comet tab's search, then the SBDB elements of each candidate ──────
    Step("1: search the comet list");
    AwaitUi(AppRig.UiTask(() => comet.SearchCometsCommand.ExecuteAsync(null)), 180_000, "comet list");
    int listed = Ui(() => comet.FilteredSearchResults.Count);
    Assert.True(listed > 0, "the SBDB comet list is empty (no network?)");
    _output.WriteLine($"comet list: {listed} comets");

    var candidates = new List<Candidate>();
    foreach (var key in Candidates)
    {
      var found = Ui(() => comet.FilteredSearchResults.FirstOrDefault(c =>
        c.PrimaryDesignation.Contains(key, StringComparison.OrdinalIgnoreCase)
        || c.Name.Contains(key, StringComparison.OrdinalIgnoreCase)));
      if (found is null)
      {
        candidates.Add(new Candidate(key, null, null, double.NaN, "not in the SBDB comet list"));
        continue;
      }
      var sbdbTask = AppRig.UiTask(() => jpl.FetchSmallBodyDataAsync(found.PrimaryDesignation));
      AwaitUi(sbdbTask, 120_000, $"SBDB {found.PrimaryDesignation}");
      var sb = sbdbTask.Result;
      if (sb is null)
      {
        candidates.Add(new Candidate(key, found, null, double.NaN, "no SBDB record"));
        continue;
      }
      double r = HeliocentricDistanceAu(sb.Q, sb.E, sb.Tp, observation);
      candidates.Add(new Candidate(key, found, sb, r,
        $"q {sb.Q:0.###} AU, e {sb.E:0.######}, Tp JD {sb.Tp:0.###}, SPK {sb.SpkId}"));
    }
    foreach (var c in candidates)
      _output.WriteLine($"  {c.Key,-11} {c.Listed?.Name.Trim() ?? "-",-40} r_h({ObservationUtc}) = {c.DistanceAu,7:0.00} AU  ({c.Note})");

    var chosen = candidates
      .Where(c => !double.IsNaN(c.DistanceAu) && Math.Abs(c.DistanceAu - TargetDistanceAu) <= DistanceToleranceAu)
      .OrderBy(c => Math.Abs(c.DistanceAu - TargetDistanceAu))
      .FirstOrDefault();
    Assert.True(chosen is not null,
      $"no candidate within ±{DistanceToleranceAu} AU of {TargetDistanceAu} AU at {ObservationUtc}: "
      + string.Join("; ", candidates.Select(c => $"{c.Key} = {c.DistanceAu:0.00} AU ({c.Note})")));
    _output.WriteLine($"chosen: {chosen!.Listed!.Name.Trim()} ({chosen.Listed.PrimaryDesignation}) at {chosen.DistanceAu:0.00} AU");

    // ── 2. the Comet tab: the comet, its SPK record, the rotation and the range ─────────────
    Step("2: select the comet and its SPK record");
    Ui(() => comet.SelectedComet = chosen.Listed);
    Assert.True(WaitFor(() => Ui(() => !comet.IsLoadingSpkRecords && comet.SpkRecords.Count > 0), 120_000), "SPK records listed");
    var record = Ui(() => comet.SpkRecords.Where(r => r.IsValid).OrderBy(r => r.EpochYear, StringComparer.Ordinal).Last());
    _output.WriteLine($"SPK records: {Ui(() => string.Join(" | ", comet.SpkRecords.Select(r => $"{r.RecordId} {r.EpochYear}")))}; using {record.RecordId}");
    Ui(() =>
    {
      comet.SelectedSpkRecord = record;
      comet.ReferenceOrbitIndex = 1; // re-osculated at the start, as the 67P review save
      comet.RotationPeriodHours = RotationPeriodHours;
      comet.Range.ObservationEpoch = ObservationUtc;
      comet.Range.ModelRotations = Rotations;
    });
    Assert.Equal(360.0 * 24.0 / RotationPeriodHours, Ui(() => comet.RotRateDegDay), 9);
    Assert.True(Ui(() => comet.Range.TryApplyModelRange()), Ui(() => comet.Range.ErrorMessage));
    _output.WriteLine($"proposed: {Ui(() => comet.Range.DisplayProposedRange)} ({Ui(() => comet.Range.ModelSpanText)})");
    AssertBox(expectedStart, Ui(() => comet.Range.StartEpoch));
    AssertBox(observation, Ui(() => comet.Range.EndEpoch));
    Assert.True(WaitFor(() => Ui(() => comet.DownloadAndCommitCommand.CanExecute(null)), 10_000),
      $"Download & Commit enabled (step {Ui(() => comet.CurrentGuidedStep)})");

    // ── 3. Download & Commit (the SPK for the range + margin), then the jets ────────────────
    Step("3: Download & Commit");
    AwaitUi(AppRig.UiTask(() => comet.DownloadAndCommitCommand.ExecuteAsync(null)), 300_000, "Download & Commit");
    _output.WriteLine($"commit: {Ui(() => comet.DownloadStatus)}");
    Assert.True(WaitFor(() => Ui(() => comet.IsAlmanacCommitted && comet.Range.HasCommittedState), 30_000),
      $"committed: {Ui(() => comet.DownloadStatus)}");
    AssertBox(expectedStart, Ui(() => comet.Range.StartEpoch));
    AssertBox(observation, Ui(() => comet.Range.EndEpoch));

    Step("3: the nucleus radius, 60 km");
    // the Comet tab's NucleusRadiusSlider binding (0.1–100 km), pushed to the engine debounced
    Ui(() => comet.ManualNucleusRadiusKmNullable = NucleusRadiusKm);
    Assert.True(WaitFor(() => CompatRadiusKm(runtime.GetCompatibilityJson()) == NucleusRadiusKm, 10_000),
      $"engine nucleus radius: {runtime.GetCompatibilityJson()}");
    Assert.Equal(NucleusRadiusKm, Ui(() => comet.ManualNucleusRadiusKm));

    Step("3: two jets in the Model tab");
    Assert.True(WaitFor(() => Ui(() => model.AddJetCommand.CanExecute(null)), 30_000), "Add jet enabled");
    Ui(() =>
    {
      // a polar and a mid-latitude source, as the 67P review save sets them up
      model.AddJetCommand.Execute(null);
      var polar = model.Jets![^1];
      polar.LatitudeDeg = -67;
      polar.LongitudeDeg = 0;
      polar.ApertureDeg = 0.2f;
      polar.StartVelocityMeanMs = 70;
      polar.StartVelocityStdMs = 6;
      model.AddJetCommand.Execute(null);
      var mid = model.Jets![^1];
      mid.LatitudeDeg = 40;
      mid.LongitudeDeg = 0;
      mid.ApertureDeg = 0.5f;
      mid.StartVelocityMeanMs = 50;
      mid.StartVelocityStdMs = 0.8f;
      // the question is about how the dust trajectories are drawn: the save opens with them on
      model.DustTrajectories = true;
    });
    var seeds = Ui(() => model.Jets!.Select(j => j.Seed).ToArray());
    Assert.Equal(2, seeds.Length);
    Assert.All(Ui(() => model.Jets!.Select(j => j.NativeError).ToArray()), e => Assert.Null(e));

    // ── 4. Run at this date: the clock to the observation date (the dust history rebuilt) ────
    Step("4: Run at this date");
    var runTask = AppRig.UiTask(() => comet.Range.RunAtAsync(null));
    AwaitUi(runTask, 300_000, "Run at this date");
    var run = runTask.Result;
    _output.WriteLine($"run: proposed {run.Proposed}, committed {run.Committed}, seeked {run.Seeked}: {run.Status}");
    Assert.True(run.Proposed && run.Committed && run.Seeked, run.Status);
    Assert.True(runtime.GetSimulationClock(out short clockC, out ulong clockNs));
    var (endC, endNs) = TimeUtils.ToTdbParts(observation);
    Assert.True(Math.Abs((TimeUtils.FromTdbParts(clockC, clockNs) - TimeUtils.FromTdbParts(endC, endNs)).TotalSeconds) < 70,
      $"clock at the observation date: {TimeUtils.FromTdbParts(clockC, clockNs):O} vs {TimeUtils.FromTdbParts(endC, endNs):O}");
    Assert.Equal(2, Ui(() => model.Jets!.Count));

    // ── 5. File › Save As .avk ──────────────────────────────────────────────────────────────
    Step("5: Save As");
    if (File.Exists(avk)) File.Delete(avk);
    Directory.CreateDirectory(Path.GetDirectoryName(avk)!);
    rig.Dialogs.Answer(avk);
    rig.Menu("File", "Save As...");
    Assert.True(WaitFor(() => File.Exists(avk) && rig.Saves.CurrentArchivePath == avk && Clean(rig.XTitle), 300_000),
      $"the .avk save, title: {rig.XTitle}");
    Assert.Equal(0, rig.Prompt.Asked);
    var saved = rig.Saves.CurrentIndex!;
    using (var zip = ZipFile.OpenRead(avk))
    {
      Assert.Equal(SaveIndex.FileName, zip.Entries[0].FullName);
      foreach (var e in zip.Entries)
        _output.WriteLine($"  {e.Length,10} {e.FullName}");
    }
    _output.WriteLine($".avk: {new FileInfo(avk).Length} bytes at {avk}");
    Assert.Single(saved.Spk);
    Assert.Equal(chosen.Sbdb!.SpkId, saved.Spk[0].NaifId);
    Assert.NotNull(saved.Spk[0].Rotation);
    Assert.Equal(360.0 * 24.0 / RotationPeriodHours, saved.Spk[0].Rotation!.RateDegDay, 6);
    Assert.Equal(Rotations, saved.Timeline.ObservationRotations);
    Assert.Equal(ObservationUtc, saved.Timeline.ObservationEpochUtc);
    Assert.NotNull(saved.Timeline.CommittedStart);
    Assert.NotNull(saved.Timeline.CommittedEnd);
    Assert.Equal(2, saved.Jets.Count);
    Assert.Equal(seeds, saved.Jets.Select(j => j.Seed).ToArray());
    Assert.True(saved.Model.DustTrajectories);
    var state = Assert.Single(saved.States);
    Assert.True(state.Latest);
    Assert.Equal(NucleusRadiusKm, CompatRadiusKm(state.CompatibilityJson));
    Assert.Equal(NucleusRadiusKm, saved.Model.ManualNucleusRadiusKm);
    var savedCompat = runtime.GetCompatibilityJson();
    var savedStart = (saved.Timeline.CommittedStart!.Centuries, saved.Timeline.CommittedStart.Nanoseconds);
    var savedEnd = (saved.Timeline.CommittedEnd!.Centuries, saved.Timeline.CommittedEnd.Nanoseconds);

    // ── 6. File › New, then the archive reopened: the same scenario, its scene restored ─────
    Step("6: File › New");
    rig.Menu("File", "New");
    Assert.True(WaitFor(() => Ui(() => !comet.IsAlmanacCommitted && (model.Jets?.Count ?? 0) == 0), 60_000), "a fresh session");
    Assert.Equal(0, rig.Prompt.Asked);

    Step("6: open the archive");
    var openTask = AppRig.UiTask(() => rig.Saves.OpenAsync(avk));
    AwaitUi(openTask, 300_000, "Open");
    var opened = openTask.Result;
    _output.WriteLine($"open: success {opened.Success}, restored {opened.StateRestored}, sought {opened.Sought}, missing {opened.MissingFiles?.Count ?? 0}");
    Assert.True(opened.Success, $"open failed: {opened.Error}");
    Assert.True(opened.StateRestored,
      $"the saved scene state was not restored\nsaved: {savedCompat}\nlive:  {runtime.GetCompatibilityJson()}");
    Assert.True(WaitFor(() => Ui(() => comet.IsAlmanacCommitted && comet.Range.HasCommittedState), 30_000), "committed after open");
    Assert.Equal(avk, rig.Saves.CurrentArchivePath);

    var reopened = rig.Saves.CurrentIndex!;
    Assert.Equal(savedStart, (reopened.Timeline.CommittedStart!.Centuries, reopened.Timeline.CommittedStart.Nanoseconds));
    Assert.Equal(savedEnd, (reopened.Timeline.CommittedEnd!.Centuries, reopened.Timeline.CommittedEnd.Nanoseconds));
    AssertBox(expectedStart, Ui(() => comet.Range.StartEpoch));
    AssertBox(observation, Ui(() => comet.Range.EndEpoch));
    Assert.Equal(RotationPeriodHours, Ui(() => comet.RotationPeriodHours), 6);
    Assert.Equal(Rotations, Ui(() => comet.Range.ModelRotations));
    Assert.Equal(ObservationUtc, Ui(() => comet.Range.ObservationEpoch));
    Assert.Equal(seeds, Ui(() => model.Jets!.Select(j => j.Seed).ToArray()));
    Assert.All(Ui(() => model.Jets!.Select(j => j.NativePsId).ToArray()), id => Assert.NotEqual(0ul, id));
    Assert.True(Ui(() => model.DustTrajectories));
    Assert.Equal(NucleusRadiusKm, Ui(() => comet.ManualNucleusRadiusKm));
    Assert.Equal(NucleusRadiusKm, CompatRadiusKm(runtime.GetCompatibilityJson()));
    Assert.True(runtime.GetSimulationClock(out clockC, out clockNs));
    Assert.True(Math.Abs((TimeUtils.FromTdbParts(clockC, clockNs) - TimeUtils.FromTdbParts(endC, endNs)).TotalSeconds) < 70,
      $"clock at the observation date after open: {TimeUtils.FromTdbParts(clockC, clockNs):O}");
    Assert.True(WaitFor(() => rig.XTitle.Contains("review_slow_rotator_2026-11", StringComparison.Ordinal) && Clean(rig.XTitle), 10_000),
      $"title after open: {rig.XTitle}");
    Assert.Equal(0, rig.Prompt.Asked);
  }

  /// The committed fixture (the archive the test above writes with <c>AETHERVK_WRITE_FIXTURE=1</c>)
  /// opened from a copy: the comet committed from the archived SPK and the SBDB elements saved
  /// with it, the range, period, rotation count, observation date, 60 km nucleus radius and jets
  /// as saved, the scene restored and the clock at the observation date. No network: the open path loads the
  /// archived kernel (<c>CometConfigService.CommitCometAsync</c>) and reads the elements from the
  /// index; nothing is fetched from the SBDB or Horizons.
  [SkippableFact]
  public void SlowRotatorFixture_OpensAndRestoresItsScene()
  {
    var fixture = AetherVk.TestAssets.TestAssets.Require("saves/review_slow_rotator_2026-11.avk");
    using var rig = new AppRig();
    // a copy: opening touches the archive's timestamps and the fixture stays as committed
    var avk = Path.Combine(rig.SavesDir, Path.GetFileName(fixture));
    File.Copy(fixture, avk);

    var observation = DateTimeOffset.ParseExact(ObservationUtc, RangeEntryViewModel.UtcBoxFormat, CultureInfo.InvariantCulture,
      DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal);
    var expectedStart = observation - TimeSpan.FromHours(Rotations * RotationPeriodHours);
    var comet = rig.Tab<CometTabViewModel>();
    var model = rig.Tab<ModelTabViewModel>();
    var runtime = rig.Host.Services.GetRequiredService<INativeRuntimeService>();

    Step("open the fixture");
    var openTask = AppRig.UiTask(() => rig.Saves.OpenAsync(avk));
    AwaitUi(openTask, 300_000, "Open");
    var opened = openTask.Result;
    _output.WriteLine($"open: success {opened.Success}, restored {opened.StateRestored}, sought {opened.Sought}, missing {opened.MissingFiles?.Count ?? 0}");
    Assert.True(opened.Success, $"open failed: {opened.Error}");
    // the fixture's state was dumped by an older engine (v9): the engine still reads it and restores it
    Assert.True(opened.StateRestored, $"the saved scene state was not restored: {runtime.GetCompatibilityJson()}");
    Assert.True(WaitFor(() => Ui(() => comet.IsAlmanacCommitted && comet.Range.HasCommittedState), 30_000), "committed after open");
    Assert.Equal(avk, rig.Saves.CurrentArchivePath);

    Step("the saved scenario");
    var index = rig.Saves.CurrentIndex!;
    var spk = Assert.Single(index.Spk);
    Assert.Equal("2014 UN271", spk.Designation);
    Assert.Equal(1003787, spk.NaifId);
    Assert.Contains("2014 UN271", Ui(() => comet.CommittedCometName));
    AssertBox(expectedStart, Ui(() => comet.Range.StartEpoch));
    AssertBox(observation, Ui(() => comet.Range.EndEpoch));
    Assert.Equal(RotationPeriodHours, Ui(() => comet.RotationPeriodHours), 6);
    Assert.Equal(Rotations, Ui(() => comet.Range.ModelRotations));
    Assert.Equal(ObservationUtc, Ui(() => comet.Range.ObservationEpoch));
    Assert.Equal(2, Ui(() => model.Jets?.Count ?? 0));
    Assert.All(Ui(() => model.Jets!.Select(j => j.NativePsId).ToArray()), id => Assert.NotEqual(0ul, id));
    Assert.All(Ui(() => model.Jets!.Select(j => j.NativeError).ToArray()), e => Assert.Null(e));
    Assert.True(Ui(() => model.DustTrajectories));
    Assert.Equal(NucleusRadiusKm, Ui(() => comet.ManualNucleusRadiusKm));
    Assert.Equal(NucleusRadiusKm, CompatRadiusKm(runtime.GetCompatibilityJson()));
    Assert.Equal(NucleusRadiusKm, CompatRadiusKm(Assert.Single(index.States).CompatibilityJson));
    Assert.True(runtime.GetSimulationClock(out short clockC, out ulong clockNs));
    var (endC, endNs) = TimeUtils.ToTdbParts(observation);
    Assert.True(Math.Abs((TimeUtils.FromTdbParts(clockC, clockNs) - TimeUtils.FromTdbParts(endC, endNs)).TotalSeconds) < 70,
      $"clock at the observation date after open: {TimeUtils.FromTdbParts(clockC, clockNs):O} vs {TimeUtils.FromTdbParts(endC, endNs):O}");
    Assert.True(WaitFor(() => rig.XTitle.Contains("review_slow_rotator_2026-11", StringComparison.Ordinal) && Clean(rig.XTitle), 10_000),
      $"title after open: {rig.XTitle}");
    Assert.Equal(0, rig.Prompt.Asked);
  }
}
