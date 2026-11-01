using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Net;
using System.Net.Http;
using System.Reactive.Concurrency;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using System.Threading;
using System.Threading.Tasks;
using AetherVk.Logic.Messages;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.Tests.Mocks;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// Reviewer feedback 2026-10-31, slide 1: the observation date apart from the generation period.
/// The Comet tab's Observation block (date UTC, N rotations before it, "Run at this date") runs one
/// chain, propose → recommit → seek to the date → set up the comparison, which the mode bar's
/// Model shares through <see cref="TimelineService.RunAtObservationAsync"/>. The SPK is not
/// downloaded again when the one on disk covers the new range; N and the date are saved.
/// </summary>
public class ObservationRunTests
{
  /// <summary>The loaded image as the chain sees it; every call is logged.</summary>
  private sealed class FakeImage(List<string> log, DateTimeOffset? utc, bool canCompare = true) : IObservationImage
  {
    public DateTimeOffset? ObservationUtc => utc;
    public bool CanSetUpComparison => canCompare;
    public string StatusMessage { get; private set; } = string.Empty;
    public Task SetUpComparisonAsync()
    {
      log.Add("compare");
      StatusMessage = "Comparison set up; the viewport is locked.";
      return Task.CompletedTask;
    }
  }

  /// <summary>Main-thread work delivered at once: the chain awaits the Comet tab's answer.</summary>
  private sealed class ImmediateSchedulers : ISchedulerProvider
  {
    public IScheduler MainThread => Scheduler.Immediate;
    public IScheduler Background => Scheduler.Immediate;
  }

  /// <summary>
  /// A standalone range entry over a real <see cref="TimelineService"/> with fake services recording
  /// the calls: the Comet tab's Download &amp; Commit is a fake recommit handler, the engine's
  /// seek a runtime mock, the image a <see cref="FakeImage"/>.
  /// </summary>
  private sealed class Rig
  {
    public readonly List<string> Log = [];
    public readonly Mock<INativeRuntimeService> Runtime = new();
    public readonly TimelineSession Session = new();
    public readonly TimelineService Timeline;
    public readonly TimeScaleService TimeScale = new(new FakeTimeScaleConverter());
    public readonly RangeEntryViewModel Range;
    public readonly List<(short C, ulong Ns)> Seeks = [];
    public bool CommitSucceeds = true;
    public int Recommits;

    public Rig(TimeScale scale = AetherVk.Logic.Services.TimeScale.Utc, double rateDegDay = 696.0, bool seekOk = true)
    {
      TimeScale.Scale = scale;
      var sessions = new Mock<ITabStateService<TimelineSession>>();
      sessions.Setup(s => s.ActiveSessionIds).Returns(new ObservableCollection<SessionId> { new SessionId(typeof(TimelineSession), 1) });
      sessions.Setup(x => x.GetSession(It.IsAny<SessionId>())).Returns(Session);
      sessions.Setup(x => x.UpdateSession(It.IsAny<SessionId>(), It.IsAny<Action<TimelineSession>>()))
        .Callback<SessionId, Action<TimelineSession>>((_, a) => a(Session));

      Runtime.Setup(r => r.SeekEpochSync(It.IsAny<short>(), It.IsAny<ulong>()))
        .Returns<short, ulong>((c, ns) => { Log.Add("seek"); Seeks.Add((c, ns)); return seekOk; });
      var config = new CometConfigService(Runtime.Object, new ImmediateSchedulers());
      Timeline = new TimelineService(Runtime.Object, new ImmediateSchedulers(), config, new BreadcrumbService());
      Range = new RangeEntryViewModel(sessions.Object, Timeline, null, () => rateDegDay, TimeScale);

      // the Comet tab's Download & Commit: commits with the proposed range
      Timeline.RecommitRequests.Subscribe(request =>
      {
        Recommits++;
        Log.Add("recommit");
        if (CommitSucceeds)
          Range.OnCommitted(Proposed());
        request.Complete(CommitSucceeds);
      });
      // a commit re-proposes the same range (the boxes follow it): one "propose" per new range
      Timeline.ProposedTimeRange.DistinctUntilChanged().Subscribe(r => { if (r is not null) Log.Add("propose"); });
      Log.Clear();
    }

    public TimeRange Proposed()
    {
      Assert.True(TimeScale.TryParse(Session.ProposedStartEpoch, TimeScale.ScaleOf(Session), out short sc, out ulong sn, out _));
      Assert.True(TimeScale.TryParse(Session.ProposedEndEpoch, TimeScale.ScaleOf(Session), out short ec, out ulong en, out _));
      return new TimeRange(sc, sn, ec, en);
    }
  }

  private static readonly DateTimeOffset MidUtc = new(2021, 12, 27, 0, 14, 47, TimeSpan.Zero);
  private const string MidUtcText = "2021-12-27T00:14:47";

  // ── the chain ──────────────────────────────────────────────────────────────

  [Fact]
  public async Task RunAtDate_ProposesCommitsSeeksAndCompares_InThatOrder()
  {
    var rig = new Rig();
    rig.Timeline.ObservationImageSource = () => new FakeImage(rig.Log, MidUtc);
    rig.Range.ModelRotations = 2;
    rig.Range.ObservationEpoch = MidUtcText;
    var statuses = new List<string>();
    rig.Range.PropertyChanged += (_, e) => { if (e.PropertyName == nameof(RangeEntryViewModel.RunStatus)) statuses.Add(rig.Range.RunStatus); };

    var run = rig.Range.RunAtDateCommand.ExecuteAsync(null);
    Assert.True(rig.Range.IsRunning); // the busy state while the chain runs
    await run;

    Assert.Equal(["propose", "recommit", "seek", "compare"], rig.Log);
    Assert.False(rig.Range.IsRunning);
    Assert.Equal("Comparison set up; the viewport is locked.", rig.Range.RunStatus);
    Assert.Contains(statuses, s => s.StartsWith("Committing", StringComparison.Ordinal));
    // the range: 2 rotations (24.8 h) ending at the date, committed
    Assert.Equal(MidUtcText, rig.Range.EndEpoch);
    Assert.Equal(MidUtcText, rig.Session.CommittedEndEpoch);
    Assert.True(TimeUtils.TryParseIso8601(rig.Range.StartEpoch, out var start));
    Assert.InRange((MidUtc - start).TotalHours - 2 * 360.0 * 24.0 / 696.0, -1.0 / 3600, 1.0 / 3600);
    Assert.False(rig.Range.IsRangeLocked);
    // the seek: the date's TDB parts (through the leap-second converter, never the UTC digits)
    Assert.Equal(FakeTimeScaleConverter.Parts(MidUtc), Assert.Single(rig.Seeks));
  }

  [Fact]
  public async Task RunAtDate_WhenTheCommitFails_StopsBeforeTheSeek()
  {
    var rig = new Rig { CommitSucceeds = false };
    rig.Timeline.ObservationImageSource = () => new FakeImage(rig.Log, MidUtc);
    rig.Range.ObservationEpoch = MidUtcText;

    var result = await rig.Range.RunAtAsync(null);

    Assert.Equal(["propose", "recommit"], rig.Log);
    Assert.Equal((true, false, false, false), (result.Proposed, result.Committed, result.Seeked, result.Compared));
    Assert.Contains("not committed", result.Status);
    Assert.Equal(result.Status, rig.Range.RunStatus);
  }

  [Fact]
  public async Task RunAtDate_WithoutAnImage_CommitsAndSeeks_ThenSaysSo()
  {
    var rig = new Rig();
    rig.Range.ObservationEpoch = MidUtcText;

    var result = await rig.Range.RunAtAsync(null);

    Assert.Equal(["propose", "recommit", "seek"], rig.Log);
    Assert.True(result.Committed && result.Seeked && !result.Compared);
    Assert.Contains(RangeEntryViewModel.NoImageText, result.Status);
  }

  [Fact]
  public async Task RunAtDate_WithAnInvalidDate_ProposesNothing()
  {
    var rig = new Rig();
    rig.Range.ObservationEpoch = "27/12/2021"; // not ISO: never read through the current culture
    var result = await rig.Range.RunAtAsync(null);
    Assert.Empty(rig.Log);
    Assert.False(result.Proposed);
    Assert.True(rig.Range.HasError);
    Assert.Equal(rig.Range.ErrorMessage, result.Status);
  }

  [Fact]
  public async Task RunAtDate_WithoutARotationPeriod_ProposesNothing()
  {
    var rig = new Rig(rateDegDay: 0);
    rig.Range.ObservationEpoch = MidUtcText;
    var result = await rig.Range.RunAtAsync(null);
    Assert.Empty(rig.Log);
    Assert.Contains("rotation period", result.Status);
  }

  /// <summary>The mode bar's Model: the same chain, with the image's date, through the timeline service.</summary>
  [Fact]
  public async Task TheModeBarsRequest_RunsTheSameChain_WithTheImageDate()
  {
    var rig = new Rig();
    rig.Timeline.ObservationImageSource = () => new FakeImage(rig.Log, MidUtc);
    rig.Range.ObservationEpoch = "2021-12-20T00:00:00"; // another date typed earlier: the request's date wins

    var result = await rig.Timeline.RunAtObservationAsync(MidUtc);

    Assert.Equal(["propose", "recommit", "seek", "compare"], rig.Log);
    Assert.True(result.Compared);
    Assert.Equal(MidUtcText, rig.Range.ObservationEpoch);
    Assert.Equal(MidUtcText, rig.Session.ObservationEpochUtc);
    Assert.Equal("Comparison set up; the viewport is locked.", result.Status);
  }

  [Fact]
  public async Task TheModeBarsRequest_WithoutACometTab_SaysSo()
  {
    var runtime = new Mock<INativeRuntimeService>();
    var config = new CometConfigService(runtime.Object, new ImmediateSchedulers());
    var timeline = new TimelineService(runtime.Object, new ImmediateSchedulers(), config, new BreadcrumbService());
    var result = await timeline.RunAtObservationAsync(MidUtc);
    Assert.Same(ObservationRunResult.NoHandler, result);
    Assert.False(await timeline.RequestRecommitAsync());
  }

  // ── the date box, "from image", the span text ──────────────────────────────

  [Fact]
  public void FromImage_TakesTheLoadedImagesDate_OrSaysThereIsNone()
  {
    var rig = new Rig();
    rig.Range.FromImageCommand.Execute(null);
    Assert.Equal(string.Empty, rig.Range.ObservationEpoch);
    Assert.Equal(RangeEntryViewModel.NoImageText, rig.Range.RunStatus);

    rig.Timeline.ObservationImageSource = () => new FakeImage(rig.Log, new DateTimeOffset(2021, 12, 27, 0, 14, 47, 345, TimeSpan.Zero));
    rig.Range.FromImageCommand.Execute(null);
    Assert.Equal(MidUtcText, rig.Range.ObservationEpoch); // UTC, whole seconds, invariant
    Assert.Equal(string.Empty, rig.Range.RunStatus);
  }

  [Fact]
  public void TheDateBox_IsUtc_WhateverTheScaleOfTheBoxes()
  {
    var rig = new Rig(AetherVk.Logic.Services.TimeScale.Tdb);
    rig.Range.ObservationEpoch = MidUtcText;
    Assert.True(rig.Range.TryApplyModelRange());
    Assert.Equal(MidUtcText, rig.Range.ObservationEpoch);
    Assert.Equal("2021-12-27T00:15:56", rig.Range.EndEpoch); // the End box in TDB: 69.184 s later
    var proposed = rig.Proposed();
    Assert.InRange(TimeUtils.FromTdbParts(proposed.EndCenturies, proposed.EndNs) - (MidUtc + FakeTimeScaleConverter.Offset), TimeSpan.FromSeconds(-1), TimeSpan.Zero);
  }

  [Fact]
  public void SpanText_IsNTimesThePeriod_FlooredAtOneDay_Invariant()
  {
    var previous = Thread.CurrentThread.CurrentCulture;
    Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("it-IT");
    try
    {
      var rig = new Rig(rateDegDay: 696.0); // P = 12.41 h
      rig.Range.ModelRotations = 3;
      Assert.Equal("= 1.55 d", rig.Range.ModelSpanText);
      rig.Range.ModelRotations = 1;
      Assert.Equal("= 1 d (minimum)", rig.Range.ModelSpanText);
      var none = new Rig(rateDegDay: 0);
      Assert.Equal(string.Empty, none.Range.ModelSpanText);
    }
    finally
    {
      Thread.CurrentThread.CurrentCulture = previous;
    }
  }

  // ── persistence ────────────────────────────────────────────────────────────

  [Fact]
  public void NRotationsAndTheDate_LiveInTheTimelineSession_AndComeBackOnReload()
  {
    var rig = new Rig();
    rig.Range.ModelRotations = 5;
    rig.Range.ObservationEpoch = MidUtcText;
    Assert.Equal(5, rig.Session.ObservationRotations);
    Assert.Equal(MidUtcText, rig.Session.ObservationEpochUtc);

    rig.Range.ModelRotations = 2;
    rig.Range.ObservationEpoch = string.Empty;
    rig.Session.ObservationRotations = 5;
    rig.Session.ObservationEpochUtc = MidUtcText;
    rig.Range.Reload(); // a save opened, Restore
    Assert.Equal((5, MidUtcText), (rig.Range.ModelRotations, rig.Range.ObservationEpoch));
  }

  /// <summary>An older save has neither field: N is the default and the date is the committed end (UTC).</summary>
  [Fact]
  public void AnOlderSave_ShowsTheDefaultN_AndTheCommittedEnd()
  {
    var rig = new Rig();
    rig.Range.ModelRotations = 7;
    rig.Session.ObservationRotations = null;
    rig.Session.ObservationEpochUtc = null;
    rig.Session.ProposedStartEpoch = "2021-12-20T00:00:00";
    rig.Session.ProposedEndEpoch = MidUtcText;
    rig.Session.CommittedStartEpoch = "2021-12-20T00:00:00";
    rig.Session.CommittedEndEpoch = MidUtcText;
    rig.Session.ProposedScale = "utc";
    rig.Range.Reload();
    Assert.Equal(RangeEntryViewModel.DefaultRotations, rig.Range.ModelRotations);
    Assert.Equal(MidUtcText, rig.Range.ObservationEpoch);
    // reloading never writes the derived values back as if typed
    Assert.Null(rig.Session.ObservationRotations);
    Assert.Null(rig.Session.ObservationEpochUtc);

    // a save typed in TDB: the committed end is shown as UTC all the same
    rig.Session.CommittedEndEpoch = "2021-12-27T00:15:56";
    rig.Session.ProposedEndEpoch = rig.Session.CommittedEndEpoch;
    rig.Session.ProposedScale = "tdb";
    rig.Range.Reload();
    Assert.Equal("2021-12-27T00:14:46", rig.Range.ObservationEpoch); // 00:15:56 TDB − 69.184 s, truncated
  }

  [Fact]
  public void TheSaveDto_RoundTripsTheFields_AndOmitsThemByDefault()
  {
    var dto = new SaveFile.TimelineDto { ObservationRotations = 4, ObservationEpochUtc = MidUtcText };
    var json = System.Text.Json.JsonSerializer.Serialize(dto);
    var back = System.Text.Json.JsonSerializer.Deserialize<SaveFile.TimelineDto>(json)!;
    Assert.Equal((4, MidUtcText), (back.ObservationRotations, back.ObservationEpochUtc));
    // an older save's JSON has no such members: null, not a default
    var old = System.Text.Json.JsonSerializer.Deserialize<SaveFile.TimelineDto>("""{"ProposedStart":"2021-12-20T00:00:00"}""")!;
    Assert.Null(old.ObservationRotations);
    Assert.Null(old.ObservationEpochUtc);
  }

  // ── the SPK is not downloaded again inside its coverage ───────────────────

  private static TimeRange Range(DateTimeOffset start, DateTimeOffset end)
  {
    var a = TimeUtils.ToTdbParts(start);
    var b = TimeUtils.ToTdbParts(end);
    return new TimeRange(a.centuries, a.nanoseconds, b.centuries, b.nanoseconds);
  }

  [Fact]
  public void SpkCoverage_IsTheRequestedWindow_AndCoversOnlyInsideIt()
  {
    string file = System.IO.Path.GetTempFileName();
    try
    {
      // downloaded for the days 2021-12-12..2021-12-28: the request widens by the margin on each side
      var c = CometTabViewModel.SpkCoverageOf("67P", "90000703", file, "2021-12-12", "2021-12-28")!;
      Assert.Equal(new DateTimeOffset(2021, 12, 12, 0, 0, 0, TimeSpan.Zero).AddDays(-HorizonJplService.SpkMarginDays), c.Start);
      Assert.Equal(new DateTimeOffset(2021, 12, 28, 0, 0, 0, TimeSpan.Zero).AddDays(HorizonJplService.SpkMarginDays), c.Stop);

      var d12 = new DateTimeOffset(2021, 12, 12, 0, 0, 0, TimeSpan.Zero);
      var d27 = new DateTimeOffset(2021, 12, 27, 0, 14, 47, TimeSpan.Zero);
      Assert.True(CometTabViewModel.SpkCovers(c, "67P", "90000703", Range(d12, d27)));
      // another observation date a week later, two rotations before it: still inside
      Assert.True(CometTabViewModel.SpkCovers(c, "67P", "90000703", Range(d27.AddDays(5), d27.AddDays(7))));
      // outside the window, another comet, another record, the file gone: a download
      Assert.False(CometTabViewModel.SpkCovers(c, "67P", "90000703", Range(d12, d27.AddDays(60))));
      Assert.False(CometTabViewModel.SpkCovers(c, "67P", "90000703", Range(d12.AddDays(-60), d27)));
      Assert.False(CometTabViewModel.SpkCovers(c, "C/2022 N2", "90000703", Range(d12, d27)));
      Assert.False(CometTabViewModel.SpkCovers(c, "67P", "90000704", Range(d12, d27)));
      // the window's last day starts at midnight: an end later that day is outside
      Assert.False(CometTabViewModel.SpkCovers(c, "67P", "90000703", Range(d12, c.Stop.AddHours(10))));
      System.IO.File.Delete(file);
      Assert.False(CometTabViewModel.SpkCovers(c, "67P", "90000703", Range(d12, d27)));
    }
    finally
    {
      if (System.IO.File.Exists(file)) System.IO.File.Delete(file);
    }
  }

  /// <summary>Records every JPL request; SBDB answers with a NAIF id, everything else fails.</summary>
  private sealed class RecordingHandler : HttpMessageHandler
  {
    public readonly List<string> Urls = [];
    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
      string url = request.RequestUri!.ToString();
      Urls.Add(url);
      if (url.Contains("sbdb.api", StringComparison.Ordinal))
        return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
        {
          Content = new StringContent("""{"object":{"spkid":"1000012","des":"67P","fullname":"67P/Churyumov-Gerasimenko"}}""")
        });
      return Task.FromResult(new HttpResponseMessage(HttpStatusCode.ServiceUnavailable));
    }
  }

  /// <summary>The Comet tab as in <c>CometTabRangeEntryTests</c>, over a recording JPL handler.</summary>
  private sealed class TabRig
  {
    public readonly TestSchedulerProvider Schedulers = new();
    public readonly Mock<INativeRuntimeService> Runtime = new();
    public readonly TimelineSession Session = new();
    public readonly RecordingHandler Http = new();
    public readonly TimelineService Timeline;
    public readonly CometTabViewModel Vm;

    public TabRig()
    {
      var translation = new Mock<ITranslationService>();
      translation.Setup(t => t.CultureChanged).Returns(Observable.Empty<System.Globalization.CultureInfo>());
      translation.Setup(t => t.GetString(It.IsAny<string>())).Returns<string>(k => k);
      var cometSessions = new Mock<ITabStateService<CometSession>>();
      cometSessions.Setup(s => s.ActiveSessionIds).Returns(new ObservableCollection<SessionId> { new SessionId(typeof(CometSession), 1) });
      cometSessions.Setup(x => x.ObserveSession(It.IsAny<SessionId>())).Returns(Observable.Return(new CometSession()));
      cometSessions.Setup(s => s.ObserveSessionList()).Returns(new BehaviorSubject<IReadOnlyList<SessionId>>(new List<SessionId>()));
      var stream = new BehaviorSubject<TimelineSession>(Session);
      var timelineSessions = new Mock<ITabStateService<TimelineSession>>();
      timelineSessions.Setup(s => s.ActiveSessionIds).Returns(new ObservableCollection<SessionId> { new SessionId(typeof(TimelineSession), 1) });
      timelineSessions.Setup(s => s.ObserveSessionList()).Returns(new BehaviorSubject<IReadOnlyList<SessionId>>(new List<SessionId>()));
      timelineSessions.Setup(x => x.GetSession(It.IsAny<SessionId>())).Returns(Session);
      timelineSessions.Setup(x => x.ObserveSession(It.IsAny<SessionId>())).Returns(stream);
      timelineSessions.Setup(x => x.UpdateSession(It.IsAny<SessionId>(), It.IsAny<Action<TimelineSession>>()))
        .Callback<SessionId, Action<TimelineSession>>((_, a) => { a(Session); stream.OnNext(Session); });
      Runtime.Setup(r => r.GetSimulationClock(out It.Ref<short>.IsAny, out It.Ref<ulong>.IsAny)).Returns(false);
      var config = new CometConfigService(Runtime.Object, Schedulers);
      var breadcrumbs = new BreadcrumbService();
      var storage = new Mock<ILocalStorageService>();
      storage.Setup(s => s.GetDownloadsPath(It.IsAny<string>())).Returns<string>(f => System.IO.Path.Combine(System.IO.Path.GetTempPath(), f));
      storage.Setup(s => s.SavePersistentAsync(It.IsAny<string>(), It.IsAny<byte[]>())).Returns(Task.CompletedTask);
      var jpl = new HorizonJplService(new ConsoleService(new Mock<IUiThreadDispatcher>().Object), breadcrumbs, storage.Object) { _httpClient = new HttpClient(Http) };
      Timeline = new TimelineService(Runtime.Object, Schedulers, config, breadcrumbs);
      Vm = new CometTabViewModel(translation.Object, Schedulers, cometSessions.Object, jpl, config, Timeline,
        breadcrumbs, storage.Object, Runtime.Object, new Mock<ITabStateService<ModelSession>>().Object,
        new ForwardingCometMessenger(), new Mock<ITabFactory>().Object, timelineSessions.Object, timeScale: new TimeScaleService(new FakeTimeScaleConverter()));
      Flush();
      Vm.FilteredSearchResults = new ObservableCollection<CometSearchResult>
      {
        new() { Name = "67P/Churyumov-Gerasimenko", PrimaryDesignation = "67P" },
      };
      Vm.SelectedComet = Vm.FilteredSearchResults[0];
      Vm.SelectedSpkRecord = new SpkRecordItem { RecordId = "90000703", Name = "67P" };
      Vm.ReferenceOrbitIndex = 0;
    }

    public void Flush()
    {
      for (int i = 0; i < 4; i++) Schedulers.MainThread.AdvanceBy(1);
    }

    /// <summary>An SPK download was asked (the records listing is also EPHEM_TYPE=SPK, but stepped over ±5 years).</summary>
    public bool SpkRequested() => Http.Urls.Exists(u => u.Contains("EPHEM_TYPE", StringComparison.Ordinal) && !u.Contains("STEP_SIZE", StringComparison.Ordinal));
  }

  [Fact]
  public async Task DownloadAndCommit_InsideTheCoverage_SkipsTheSpkDownload()
  {
    string file = System.IO.Path.GetTempFileName();
    try
    {
      var rig = new TabRig();
      rig.Vm.NoteSpkCoverage(CometTabViewModel.SpkCoverageOf("67P", "90000703", file, "2021-12-12", "2021-12-28"));
      rig.Vm.Range.StartEpoch = "2021-12-20T00:00:00";
      rig.Vm.Range.EndEpoch = "2021-12-27T00:14:47";
      rig.Flush();
      Assert.True(rig.Vm.DownloadAndCommitCommand.CanExecute(null));

      await rig.Vm.DownloadAndCommitCommand.ExecuteAsync(null);

      Assert.False(rig.SpkRequested(), string.Join("\n", rig.Http.Urls));
      // the commit went on past the download with the file on disk (the engine mock refuses the range)
      Assert.Equal("Timeline could not be applied: commit cancelled.", rig.Vm.DownloadStatus);
      rig.Runtime.Verify(r => r.SetEpochRange(It.IsAny<short>(), It.IsAny<ulong>(), It.IsAny<short>(), It.IsAny<ulong>()), Times.Once);

      // a range past the window: the download again (here refused by the offline handler)
      rig.Vm.Range.EndEpoch = "2022-03-01T00:00:00";
      rig.Flush();
      await rig.Vm.DownloadAndCommitCommand.ExecuteAsync(null);
      Assert.True(rig.SpkRequested());
      Assert.Equal("SPK download failed.", rig.Vm.DownloadStatus);
    }
    finally
    {
      if (System.IO.File.Exists(file)) System.IO.File.Delete(file);
    }
  }
}
