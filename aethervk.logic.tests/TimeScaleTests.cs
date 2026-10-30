using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using System.Threading;
using System.Threading.Tasks;
using AetherVk.Logic.Input;
using AetherVk.Logic.Messages;
using AetherVk.Logic.Models;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using AetherVk.Logic.Tests.Mocks;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using CommunityToolkit.Mvvm.Messaging;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// Dates typed and shown in UTC or TDB (Settings › Time scale for dates). The engine, the committed
/// range and the save hold TDB parts; a UTC text goes through <see cref="ITimeScaleConverter"/>
/// (here the deterministic <see cref="FakeTimeScaleConverter"/>: TDB = UTC + 69.184 s). The FITS
/// mid-exposure is UTC, so the "Model range" end must be converted (69 s on the reviewer's image).
/// </summary>
public class TimeScaleTests
{
  private static readonly DateTimeOffset MidUtc = new(2021, 12, 27, 0, 14, 47, TimeSpan.Zero);
  private const string MidUtcText = "2021-12-27T00:14:47";
  private const string MidTdbText = "2021-12-27T00:15:56.184";

  /// <summary>No network in tests: every JPL request fails at once.</summary>
  private sealed class OfflineHandler : HttpMessageHandler
  {
    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
      Task.FromResult(new HttpResponseMessage(HttpStatusCode.ServiceUnavailable));
  }

  /// <summary>The Comet and Timeline tabs over one session, one messenger and one time-scale service (UTC by default).</summary>
  private sealed class Rig
  {
    public readonly TestSchedulerProvider Schedulers = new();
    public readonly Mock<INativeRuntimeService> Runtime = new();
    public readonly TimelineSession Session;
    public readonly ForwardingCometMessenger Messenger = new();
    public readonly FakeTimeScaleConverter Fake = new();
    public readonly TimeScaleService TimeScale;
    public readonly TimelineService Timeline;
    public readonly CometTabViewModel Vm;
    public readonly TimelineTabViewModel Tl;
    public (short C, ulong N) Clock;

    public Rig(TimelineSession? session = null, PreferencesService? prefs = null, (short C, ulong N)? clock = null)
    {
      Session = session ?? new TimelineSession();
      TimeScale = new TimeScaleService(Fake, prefs) { Now = () => Fake.Now };
      if (prefs is null) TimeScale.Scale = AetherVk.Logic.Services.TimeScale.Utc;

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

      Clock = clock ?? TimeUtils.ToTdbParts(new DateTimeOffset(2021, 12, 20, 0, 0, 0, TimeSpan.Zero));
      Runtime.Setup(r => r.PauseSimulationSync()).Returns(true);
      Runtime.Setup(r => r.GetSimulationClock(out It.Ref<short>.IsAny, out It.Ref<ulong>.IsAny))
        .Returns(new ClockReader((out short c, out ulong ns) => { (c, ns) = Clock; return true; }));
      Runtime.Setup(r => r.SeekEpochSync(It.IsAny<short>(), It.IsAny<ulong>()))
        .Returns<short, ulong>((c, ns) => { Clock = (c, ns); return true; });

      var config = new CometConfigService(Runtime.Object, Schedulers);
      var breadcrumbs = new BreadcrumbService();
      var storage = new Mock<ILocalStorageService>();
      var jpl = new HorizonJplService(new ConsoleService(new Mock<IUiThreadDispatcher>().Object), breadcrumbs, storage.Object) { _httpClient = new HttpClient(new OfflineHandler()) };
      Timeline = new TimelineService(Runtime.Object, Schedulers, config, breadcrumbs);

      Vm = new CometTabViewModel(translation.Object, Schedulers, cometSessions.Object, jpl, config, Timeline,
        breadcrumbs, storage.Object, Runtime.Object, new Mock<ITabStateService<ModelSession>>().Object,
        Messenger, new Mock<ITabFactory>().Object, timelineSessions.Object, TimeScale);
      Tl = new TimelineTabViewModel(translation.Object, Schedulers, timelineSessions.Object, cometSessions.Object,
        Timeline, Runtime.Object, Messenger, TimeScale);
      Tl.CurrentSession = Session;
      Flush();
    }

    private delegate bool ClockReader(out short centuries, out ulong nanoseconds);

    /// <summary>Delivers the main-thread work (a proposal hops the main-thread scheduler twice).</summary>
    public void Flush()
    {
      for (int i = 0; i < 4; i++) Schedulers.MainThread.AdvanceBy(1);
    }
  }

  private static double Seconds((short C, ulong N) a, (short C, ulong N) b) =>
    (TimeUtils.FromTdbParts(a.C, a.N) - TimeUtils.FromTdbParts(b.C, b.N)).TotalSeconds;

  // ── the service ────────────────────────────────────────────────────────────

  [Fact]
  public void Service_ParsesAndFormatsInTheChosenScale()
  {
    var fake = new FakeTimeScaleConverter();
    var svc = new TimeScaleService(fake) { Now = () => fake.Now };
    Assert.Equal(TimeScale.Utc, svc.Scale); // the default without a preference file
    Assert.Equal("UTC", svc.ScaleName);

    Assert.True(svc.TryParse(MidUtcText, out short c, out ulong ns, out var status));
    Assert.Equal(UtcStatus.Ok, status);
    Assert.Equal(FakeTimeScaleConverter.Parts(MidUtc), (c, ns));
    Assert.Equal(69.184, Seconds((c, ns), TimeUtils.ToTdbParts(MidUtc)), 6);
    Assert.Equal(MidUtcText, svc.Format(c, ns, TimeScaleService.IsoSecondsFormat));
    Assert.Equal(MidTdbText, svc.Format(c, ns, TimeScaleService.IsoSecondsFormat + ".FFF", TimeScale.Tdb));

    // the same text read as TDB is 69.184 s earlier (plain parts)
    Assert.True(svc.TryParse(MidUtcText, TimeScale.Tdb, out short tc, out ulong tn, out _));
    Assert.Equal(TimeUtils.ToTdbParts(MidUtc), (tc, tn));

    // reformatting keeps the instant and the trailing Z, adds the fraction only when there is one
    Assert.Equal(MidTdbText, svc.Reformat(MidUtcText, TimeScale.Utc, TimeScale.Tdb));
    Assert.Equal(MidTdbText + "Z", svc.Reformat(MidUtcText + "Z", TimeScale.Utc, TimeScale.Tdb));
    Assert.Equal(MidUtcText, svc.Reformat(MidTdbText, TimeScale.Tdb, TimeScale.Utc));
    Assert.Equal("garbage", svc.Reformat("garbage", TimeScale.Utc, TimeScale.Tdb));

    Assert.Equal("TDB = UTC + 69.18 s", svc.OffsetText(c, ns));
    Assert.Contains("TDB = UTC + 69.18 s", svc.ScaleTip);
    Assert.Equal("Leap seconds: naif0012.tls, last +37 s on 2017-01-01", svc.LeapSecondsSummary);
    Assert.Contains("Leap seconds: naif0012.tls, last +37 s on 2017-01-01", svc.HelpText);
    Assert.Equal(new DateTime(2027, 4, 30, 0, 0, 0, DateTimeKind.Utc), svc.ValidityEnd);

    svc.Scale = TimeScale.Tdb;
    Assert.Equal("TDB", svc.ScaleName);
    Assert.True(svc.TryParse(MidTdbText, out c, out ns, out status));
    Assert.Equal(UtcStatus.Ok, status);
    Assert.Equal(FakeTimeScaleConverter.Parts(MidUtc), (c, ns));
    Assert.Equal(UtcStatus.Ok, svc.StatusOf("2030-01-01T00:00:00")); // TDB never warns
  }

  /// <summary>The 2026-10-30 e2e: a seek to 00:00:00 UTC landed a fraction of a µs short (the sim
  /// clock counts whole µs) and the box read 23:59:59. A text without a fraction shows the nearest
  /// ms; one with a fraction keeps the exact instant.</summary>
  [Fact]
  public void Service_AHairShortOfASecond_ShowsThatSecond()
  {
    var fake = new FakeTimeScaleConverter();
    var svc = new TimeScaleService(fake) { Now = () => fake.Now };
    var (c, ns) = FakeTimeScaleConverter.Parts(MidUtc);
    Assert.Equal(MidUtcText, svc.Format(c, ns - 700, TimeScaleService.IsoSecondsFormat));
    Assert.Equal("2021-12-27T00:15:56", svc.Format(c, ns - 700, TimeScaleService.IsoSecondsFormat, TimeScale.Tdb));
    Assert.NotEqual(MidUtcText, svc.Format(c, ns - 700, TimeScaleService.IsoSecondsFormat + ".fffffff"));
    // across a century: the carry
    Assert.Equal(((short)1, 499_999UL), TimeScaleService.HalfMsLater(0, 36525UL * 86400UL * 1_000_000_000UL - 1));
  }

  [Fact]
  public void Service_AHairShortOfASecond_WithASubMsOffset_ShowsThatSecond()
  {
    // the real TDB - UTC on 2021-12-14 is 69.1834 s: the seek lands up to 1 us short of the typed
    // UTC second; the text must still read that second (e2e SeekWhilePaused read 23:59:59)
    var fake = new FakeTimeScaleConverter { UtcOffset = TimeSpan.FromTicks(691_834_128) };
    var svc = new TimeScaleService(fake) { Now = () => fake.Now };
    var second = new DateTimeOffset(2021, 12, 14, 0, 0, 0, TimeSpan.Zero);
    var (c, ns) = TimeUtils.ToTdbParts(second + fake.UtcOffset);
    Assert.Equal("2021-12-14T00:00:00", svc.Format(c, ns - 1_000, TimeScaleService.IsoSecondsFormat));
    Assert.Equal("2021-12-14T00:00:00", svc.Format(c, ns, TimeScaleService.IsoSecondsFormat));
    // a whole ms short is a different instant: the second before
    Assert.Equal("2021-12-13T23:59:59", svc.Format(c, ns - 1_000_000, TimeScaleService.IsoSecondsFormat));
  }

  [Fact]
  public void Service_WithoutAConverter_IsTdbOnly()
  {
    var svc = TimeScaleService.TdbOnly();
    Assert.Equal(TimeScale.Tdb, svc.Scale);
    Assert.True(svc.TryParse(MidUtcText, out short c, out ulong ns, out _));
    Assert.Equal(TimeUtils.ToTdbParts(MidUtc), (c, ns));
    Assert.Equal(string.Empty, svc.OffsetText());
    Assert.Equal(MidUtcText, svc.Reformat(MidUtcText, TimeScale.Utc, TimeScale.Tdb));
  }

  [Fact]
  public void Service_WithoutAConverter_NeverReadsUtcAsTdb()
  {
    var svc = TimeScaleService.TdbOnly();
    // a UTC text is refused, not taken as TDB digits (69 s off)
    Assert.False(svc.TryParse(MidUtcText, TimeScale.Utc, out _, out _, out var status));
    Assert.Equal(UtcStatus.Malformed, status);
    Assert.Throws<InvalidOperationException>(() => svc.FromUtc(MidUtc));
    // a UTC display it cannot give is labelled TDB
    var (c, ns) = TimeUtils.ToTdbParts(MidUtc);
    Assert.EndsWith(" TDB", svc.Format(c, ns, TimeScaleService.IsoSecondsFormat, TimeScale.Utc));
  }

  [Fact]
  public void Service_StatusOutsideTheLeapSecondTable_WarnsButConverts()
  {
    var fake = new FakeTimeScaleConverter();
    var svc = new TimeScaleService(fake) { Now = () => fake.Now };
    Assert.True(svc.TryParse("2028-01-01T00:00:00", out short c, out ulong ns, out var status));
    Assert.Equal(UtcStatus.BeyondLskValidity, status);
    Assert.Equal(FakeTimeScaleConverter.Parts(new DateTimeOffset(2028, 1, 1, 0, 0, 0, TimeSpan.Zero)), (c, ns));
    Assert.True(TimeScaleService.Warns(status));
    Assert.Contains("2027-04-30", svc.WarningText(status));
    Assert.Contains("2017-01-01", svc.WarningText(status));

    Assert.True(svc.TryParse("1969-07-20T20:17:40", out _, out _, out status));
    Assert.Equal(UtcStatus.BeforeLeapSeconds, status);
    Assert.Contains("1972", svc.WarningText(status));

    Assert.Equal(UtcStatus.Ok, svc.StatusOf(MidUtcText));
    Assert.Null(svc.WarningText(UtcStatus.Ok));
    Assert.False(svc.TryParse("2021-02-30T00:00:00", out _, out _, out _)); // no date at all
  }

  // ── the range boxes ────────────────────────────────────────────────────────

  [Fact]
  public void RangeEntry_UtcText_ProposesTheConvertedParts()
  {
    var rig = new Rig();
    Assert.Equal("UTC", rig.Vm.Range.ScaleName);
    rig.Vm.Range.StartEpoch = "2021-12-17T00:00:00";
    rig.Vm.Range.EndEpoch = MidUtcText;
    rig.Flush();

    Assert.False(rig.Vm.Range.HasError);
    var proposal = rig.Vm.ProposedRangeForCommit!;
    Assert.Equal(FakeTimeScaleConverter.Parts(MidUtc), (proposal.EndCenturies, proposal.EndNs));
    Assert.Equal(FakeTimeScaleConverter.Parts(new DateTimeOffset(2021, 12, 17, 0, 0, 0, TimeSpan.Zero)), (proposal.StartCenturies, proposal.StartNs));
    Assert.Equal("utc", rig.Session.ProposedScale);
    Assert.Equal(MidUtcText, rig.Session.ProposedEndEpoch);
    Assert.False(rig.Vm.Range.HasTimeScaleWarning);
  }

  [Fact]
  public void SwitchingTheScale_ReformatsTheBoxes_WithoutChangingTheProposal()
  {
    var rig = new Rig();
    rig.Vm.Range.StartEpoch = "2021-12-17T00:00:00";
    rig.Vm.Range.EndEpoch = MidUtcText;
    rig.Flush();
    var before = rig.Vm.ProposedRangeForCommit!;

    rig.TimeScale.Scale = TimeScale.Tdb;
    rig.Flush();
    Assert.Equal("TDB", rig.Vm.Range.ScaleName);
    Assert.Equal(MidTdbText, rig.Vm.Range.EndEpoch);
    Assert.Equal("2021-12-17T00:01:09.184", rig.Vm.Range.StartEpoch);
    Assert.Equal("tdb", rig.Session.ProposedScale);
    Assert.Equal(MidTdbText, rig.Session.ProposedEndEpoch);
    Assert.Equal(before, rig.Vm.ProposedRangeForCommit);
    Assert.False(rig.Vm.Range.HasError);

    // and back: the typed UTC text again
    rig.TimeScale.Scale = TimeScale.Utc;
    rig.Flush();
    Assert.Equal(MidUtcText, rig.Vm.Range.EndEpoch);
    Assert.Equal("2021-12-17T00:00:00", rig.Vm.Range.StartEpoch);
    Assert.Equal(before, rig.Vm.ProposedRangeForCommit);
  }

  [Fact]
  public void SwitchingTheScale_ReformatsTheCommittedRange_AndKeepsItsParts()
  {
    var rig = new Rig();
    rig.Vm.Range.StartEpoch = "2021-12-17T00:00:00";
    rig.Vm.Range.EndEpoch = MidUtcText;
    rig.Flush();
    var range = rig.Vm.ProposedRangeForCommit!;
    rig.Vm.IsAlmanacCommitted = true;
    rig.Vm.Range.OnCommitted(range);
    rig.Messenger.Send(new CometCommittedMessage(range));
    rig.Flush();
    Assert.Equal(MidUtcText, rig.Session.CommittedEndEpoch);
    Assert.Equal(MidUtcText, rig.Tl.EndEpoch);

    rig.TimeScale.Scale = TimeScale.Tdb;
    rig.Flush();
    Assert.Equal(MidTdbText, rig.Session.CommittedEndEpoch);
    Assert.Equal(MidTdbText, rig.Tl.EndEpoch);
    Assert.Equal(MidTdbText, rig.Vm.Range.EndEpoch);
    // the read-only Timeline boxes parse back to the same TDB parts
    Assert.True(rig.TimeScale.TryParse(rig.Session.CommittedEndEpoch, out short c, out ulong ns, out _));
    Assert.Equal((range.EndCenturies, range.EndNs), (c, ns));
  }

  // ── the FITS image's "Model range" ─────────────────────────────────────────

  [Fact]
  public void FitsModelRange_ConvertsTheMidExposureUtc_ToTdbParts()
  {
    var rig = new Rig();
    const double rateDegDay = 696.2127316680096; // the reviewer's save: P ≈ 12.41 h
    rig.Vm.RotRateDegDay = rateDegDay;
    rig.Vm.Range.ModelRotations = 2;
    int calls = rig.Fake.UtcToTdbCalls;
    rig.Timeline.RequestModelRange(MidUtc);
    rig.Flush();

    Assert.True(rig.Fake.UtcToTdbCalls > calls, "the request did not go through the converter");
    Assert.Equal(MidUtcText, rig.Vm.Range.ModelEndEpoch); // UTC mode: the image's own DATE-OBS text
    Assert.Equal(MidUtcText, rig.Vm.Range.EndEpoch);
    Assert.False(rig.Vm.Range.HasError);
    var proposal = rig.Vm.ProposedRangeForCommit!;
    Assert.Equal(FakeTimeScaleConverter.Parts(MidUtc), (proposal.EndCenturies, proposal.EndNs));
    Assert.NotEqual(TimeUtils.ToTdbParts(MidUtc), (proposal.EndCenturies, proposal.EndNs)); // the 69 s bug
    double periodHours = 360.0 * 24.0 / rateDegDay;
    double spanH = Seconds((proposal.EndCenturies, proposal.EndNs), (proposal.StartCenturies, proposal.StartNs)) / 3600;
    Assert.InRange(spanH - Math.Max(2 * periodHours, 24 * RangeEntryViewModel.MinRangeDays), -1.0 / 3600, 1.0 / 3600);
  }

  [Fact]
  public void FitsModelRange_InTdbMode_ShowsTheMidExposureAsTdb()
  {
    var rig = new Rig();
    rig.TimeScale.Scale = TimeScale.Tdb;
    rig.Vm.RotRateDegDay = 696.0;
    rig.Vm.Range.ModelRotations = 2;
    rig.Timeline.RequestModelRange(MidUtc);
    rig.Flush();

    Assert.Equal("2021-12-27T00:15:56", rig.Vm.Range.ModelEndEpoch); // the box format has no fraction
    Assert.False(rig.Vm.Range.HasError);
    var proposal = rig.Vm.ProposedRangeForCommit!;
    // the boxes hold the second: the proposal is the box's text, 184 ms before the converted instant
    Assert.InRange(Seconds(FakeTimeScaleConverter.Parts(MidUtc), (proposal.EndCenturies, proposal.EndNs)), 0, 1);
  }

  // ── beyond the leap-second table ───────────────────────────────────────────

  [Fact]
  public void RangeEntry_BeyondTheLeapSecondTable_Warns_AndUseTdbSwitchesTheScale()
  {
    var rig = new Rig();
    rig.Vm.Range.StartEpoch = "2027-12-01T00:00:00";
    rig.Vm.Range.EndEpoch = "2028-01-01T00:00:00";
    rig.Flush();

    // never refused: proposed, converted assuming no further leap second
    Assert.False(rig.Vm.Range.HasError);
    Assert.NotNull(rig.Vm.ProposedRangeForCommit);
    Assert.True(rig.Vm.Range.HasTimeScaleWarning);
    Assert.Equal(UtcStatus.BeyondLskValidity, rig.Vm.Range.TimeScaleStatus);
    Assert.Contains("leap second", rig.Vm.Range.TimeScaleWarningText);

    Assert.True(rig.Vm.Range.UseTdbCommand.CanExecute(null));
    rig.Vm.Range.UseTdbCommand.Execute(null);
    rig.Flush();
    Assert.Equal(TimeScale.Tdb, rig.TimeScale.Scale);
    Assert.False(rig.Vm.Range.HasTimeScaleWarning);
    Assert.Equal(UtcStatus.Ok, rig.Vm.Range.TimeScaleStatus);
    Assert.Equal("2028-01-01T00:01:09.184", rig.Vm.Range.EndEpoch);
    Assert.NotNull(rig.Vm.ProposedRangeForCommit);
  }

  [Fact]
  public void RangeEntry_Before1972_Warns()
  {
    var rig = new Rig();
    rig.Vm.Range.StartEpoch = "1969-07-16T13:32:00";
    rig.Vm.Range.EndEpoch = "1969-07-24T16:50:35";
    rig.Flush();
    Assert.False(rig.Vm.Range.HasError);
    Assert.Equal(UtcStatus.BeforeLeapSeconds, rig.Vm.Range.TimeScaleStatus);
    Assert.True(rig.Vm.Range.HasTimeScaleWarning);
  }

  [Fact]
  public async Task TimelineEpochBox_Utc_SeeksTheConvertedInstant_AndWarnsBeyondTheTable()
  {
    var session = new TimelineSession { CommittedStartEpoch = "2027-01-01T00:00:00Z", CommittedEndEpoch = "2028-06-01T00:00:00Z" };
    var rig = new Rig(session);
    Assert.Equal("utc", session.ProposedScale); // the texts are pinned to the chosen scale
    Assert.Equal("UTC", rig.Tl.ScaleName);

    rig.Tl.IsEditingCurrentEpoch = true;
    rig.Tl.CurrentEpochText = "2028-01-01 00:00:00";
    await rig.Tl.SeekToEpochCommand.ExecuteAsync(null);

    var target = FakeTimeScaleConverter.Parts(new DateTimeOffset(2028, 1, 1, 0, 0, 0, TimeSpan.Zero));
    rig.Runtime.Verify(r => r.SeekEpochSync(target.Centuries, target.Nanoseconds), Times.Once);
    Assert.False(rig.Tl.HasCurrentEpochError);
    Assert.Equal("2028-01-01 00:00:00", rig.Tl.CurrentEpochText); // shown back in UTC
    Assert.Equal("2028-01-01 00:00 (UTC)", session.CurrentEpochString);
    Assert.True(rig.Tl.HasCurrentEpochTimeScaleWarning);
    Assert.Contains("leap second", rig.Tl.CurrentEpochTimeScaleWarningText);

    rig.Tl.UseTdbCommand.Execute(null);
    rig.Flush();
    Assert.Equal(TimeScale.Tdb, rig.TimeScale.Scale);
    Assert.False(rig.Tl.HasCurrentEpochTimeScaleWarning);
    Assert.Equal("TDB", rig.Tl.ScaleName);
    Assert.Equal("2027-01-01T00:01:09.184Z", session.CommittedStartEpoch); // the read-only boxes follow
  }

  [Fact]
  public void TimelineEpochBox_Utc_ShowsTheClockInUtc_WithTheSuffix()
  {
    var session = new TimelineSession { CommittedStartEpoch = "2021-12-17T00:00:00", CommittedEndEpoch = "2021-12-28T00:00:00" };
    // the clock is polled at construction, on a seek and while playing (not on a timer when paused)
    var rig = new Rig(session, clock: FakeTimeScaleConverter.Parts(MidUtc));
    Assert.Equal("2021-12-27 00:14:47", rig.Tl.CurrentEpochText);
    Assert.Equal("2021-12-27 00:14 (UTC)", session.CurrentEpochString);
    Assert.False(rig.Tl.HasCurrentEpochTimeScaleWarning);

    Assert.Equal("2021-12-27 00:15 (TDB)", TimelineTabViewModel.ClockLabel(rig.Clock.C, rig.Clock.N, "2021-12-17T00:00:00"));
    Assert.Equal("2021-12-27 00:14 (UTC)", TimelineTabViewModel.ClockLabel(rig.Clock.C, rig.Clock.N, session.CommittedStartEpoch, rig.TimeScale, TimeScale.Tdb));
  }

  // ── preferences ────────────────────────────────────────────────────────────

  [Fact]
  public void Preferences_TimeScale_DefaultsToUtc_AndRoundTrips()
  {
    var dir = Path.Combine(Path.GetTempPath(), "avk_timescale_" + Guid.NewGuid().ToString("N"));
    Directory.CreateDirectory(dir);
    try
    {
      var path = Path.Combine(dir, "preferences.json");
      var prefs = new PreferencesService(path, new InputRegistry());
      Assert.Equal(TimeScale.Utc, prefs.TimeScale);

      var fake = new FakeTimeScaleConverter();
      var svc = new TimeScaleService(fake, prefs);
      Assert.Equal(TimeScale.Utc, svc.Scale);
      int changes = 0;
      svc.ScaleChanged += _ => changes++;
      svc.Scale = TimeScale.Tdb;
      Assert.Equal(1, changes);
      svc.Scale = TimeScale.Tdb; // no event when unchanged
      Assert.Equal(1, changes);
      Assert.Equal(TimeScale.Tdb, prefs.TimeScale);
      Assert.Contains("\"time_scale\"", File.ReadAllText(path));
      Assert.Contains("\"tdb\"", File.ReadAllText(path));

      var reloaded = new PreferencesService(path, new InputRegistry());
      reloaded.Load(); // as App startup does
      Assert.Equal(TimeScale.Tdb, reloaded.TimeScale);
      Assert.Equal(TimeScale.Tdb, new TimeScaleService(fake, reloaded).Scale);

      // an unknown value falls back
      Assert.Equal(TimeScale.Utc, TimeScaleService.Parse("tai", TimeScale.Utc));
      Assert.Equal(TimeScale.Tdb, TimeScaleService.Parse(null, TimeScale.Tdb));
      Assert.Equal(TimeScale.Utc, TimeScaleService.Parse("UTC", TimeScale.Tdb));
    }
    finally
    {
      Directory.Delete(dir, recursive: true);
    }
  }

  // ── the save file ──────────────────────────────────────────────────────────

  private static AppSaveFileHost HostWith(AppSaveFileHostTests.Rig rig, TimeScaleService timeScale) =>
    new(rig.Runtime.Object, AppSaveFileHostTests.Rig.Sessions(rig.Comet).Object,
      AppSaveFileHostTests.Rig.Sessions(rig.Timeline).Object, AppSaveFileHostTests.Rig.Sessions(rig.Model).Object,
      rig.CometConfig, rig.TimelineService, rig.Assets, rig.Billboards, new CameraServiceRegistry(), rig.Messenger,
      new BreadcrumbService(), rig.Schedulers, timeScale);

  [Fact]
  public void Save_OldTimelineWithoutAScale_IsReadAsTdb()
  {
    var rig = new AppSaveFileHostTests.Rig();
    var fake = new FakeTimeScaleConverter();
    var host = HostWith(rig, new TimeScaleService(fake) { Now = () => fake.Now });

    // the reviewer's save: committed parts, texts typed in TDB, no proposed_scale
    var start = TimeUtils.ToTdbParts(new DateTimeOffset(2021, 12, 17, 15, 48, 0, TimeSpan.Zero));
    var end = TimeUtils.ToTdbParts(new DateTimeOffset(2021, 12, 28, 0, 0, 0, TimeSpan.Zero));
    host.ApplyTimeline(new TimelineDto
    {
      ProposedStart = "2021-12-17T15:48:00", ProposedEnd = "2021-12-28T00:00:00",
      CommittedStart = new EpochParts(start.centuries, start.nanoseconds), CommittedEnd = new EpochParts(end.centuries, end.nanoseconds),
    }, committed: true);

    Assert.Equal("tdb", rig.Timeline.ProposedScale);
    Assert.Equal("2021-12-17T15:48:00", rig.Timeline.ProposedStartEpoch);
    // the texts name the committed parts in TDB: kept as they are
    Assert.Equal("2021-12-17T15:48:00", rig.Timeline.CommittedStartEpoch);
    Assert.Equal("2021-12-28T00:00:00", rig.Timeline.CommittedEndEpoch);

    // captured back with the scale pinned
    var captured = host.Capture().Timeline;
    Assert.Equal("tdb", captured.ProposedScale);
    Assert.Equal("2021-12-17T15:48:00", captured.ProposedStart);
  }

  [Fact]
  public void Save_UtcTexts_AreReadInUtc_AndCapturedWithTheirScale()
  {
    var rig = new AppSaveFileHostTests.Rig();
    var fake = new FakeTimeScaleConverter();
    var host = HostWith(rig, new TimeScaleService(fake) { Now = () => fake.Now });

    var end = FakeTimeScaleConverter.Parts(MidUtc);
    var start = FakeTimeScaleConverter.Parts(new DateTimeOffset(2021, 12, 17, 0, 0, 0, TimeSpan.Zero));
    host.ApplyTimeline(new TimelineDto
    {
      ProposedStart = "2021-12-17T00:00:00", ProposedEnd = MidUtcText, ProposedScale = "utc",
      CommittedStart = new EpochParts(start.Centuries, start.Nanoseconds), CommittedEnd = new EpochParts(end.Centuries, end.Nanoseconds),
    }, committed: true);

    Assert.Equal("utc", rig.Timeline.ProposedScale);
    Assert.Equal(MidUtcText, rig.Timeline.CommittedEndEpoch); // the same instant in UTC: the typed text
    TimeRange? proposed = null;
    using (rig.TimelineService.ProposedTimeRange.Subscribe(r => proposed = r))
      rig.Schedulers.MainThread.AdvanceBy(4);
    Assert.NotNull(proposed);
    Assert.Equal((end.Centuries, end.Nanoseconds), (proposed!.EndCenturies, proposed.EndNs));

    var captured = host.Capture().Timeline;
    Assert.Equal("utc", captured.ProposedScale);
    Assert.Equal(MidUtcText, captured.ProposedEnd);
    Assert.Equal("2021-12-17T00:00:00", captured.ProposedStart); // the committed parts come from the comet commit, not this rig
  }

  [Fact]
  public void Save_FreshSession_IsCapturedInTheChosenScale()
  {
    var rig = new AppSaveFileHostTests.Rig();
    var svc = new TimeScaleService(new FakeTimeScaleConverter());
    svc.Scale = TimeScale.Utc;
    var host = HostWith(rig, svc);
    Assert.Equal(string.Empty, rig.Timeline.ProposedScale);
    Assert.Equal("utc", host.Capture().Timeline.ProposedScale);
    svc.Scale = TimeScale.Tdb;
    Assert.Equal("tdb", host.Capture().Timeline.ProposedScale);
  }

  [Fact]
  public void SaveIndex_ProposedScale_IsSerializedAsSnakeCase()
  {
    var index = new SaveIndex { Timeline = new TimelineDto { ProposedStart = "2021-12-17T00:00:00", ProposedScale = "utc" } };
    string json = System.Text.Json.JsonSerializer.Serialize(index, SaveIndexJsonContext.Default.SaveIndex);
    Assert.Contains("\"proposed_scale\"", json);
    var back = System.Text.Json.JsonSerializer.Deserialize(json, SaveIndexJsonContext.Default.SaveIndex);
    Assert.Equal("utc", back!.Timeline.ProposedScale);
    // an old index has none
    var old = System.Text.Json.JsonSerializer.Deserialize("{\"timeline\":{\"proposed_start\":\"2021-12-17T00:00:00\"}}", SaveIndexJsonContext.Default.SaveIndex);
    Assert.Null(old!.Timeline.ProposedScale);
    Assert.Equal(TimeScale.Tdb, TimeScaleService.Parse(old.Timeline.ProposedScale, TimeScale.Tdb));
  }

  // ── TimeUtils ──────────────────────────────────────────────────────────────

  [Fact]
  public void TimeUtils_TdbParts_AreTheEngineScale_NoOffset()
  {
    // J2000 TDB = 2000-01-01T12:00:00 of the calendar the parts count from
    var (c, ns) = TimeUtils.ToTdbParts(new DateTimeOffset(2000, 1, 1, 12, 0, 0, TimeSpan.Zero));
    Assert.Equal((short)0, c);
    Assert.Equal(0ul, ns);
    Assert.Equal("2000-01-01 12:00", TimeUtils.FormatTdbEpoch(0, 0));
    Assert.Equal("2000-01-01T12:00:00", TimeUtils.FormatTdbEpoch(0, 0, "yyyy-MM-dd'T'HH:mm:ss"));
    var parts = TimeUtils.ToTdbParts(MidUtc);
    Assert.Equal(MidUtc, TimeUtils.FromTdbParts(parts.centuries, parts.nanoseconds));
  }
}
