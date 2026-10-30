using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Net;
using System.Net.Http;
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
using CommunityToolkit.Mvvm.Messaging;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// Reviewer feedback item 7 ("enter the dates here and make the timeline only show progress"):
/// the range is entered in the Comet tab, the Timeline tab shows the committed range read-only.
/// Item 3 ("hard to understand which parameters to enter"): a numbered step strip driven by the
/// tab's state. Both tabs run over one timeline session and one messenger, as in the app.
/// </summary>
public class CometTabRangeEntryTests
{
  /// <summary>No network in tests: every JPL request fails at once.</summary>
  private sealed class OfflineHandler : HttpMessageHandler
  {
    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
      Task.FromResult(new HttpResponseMessage(HttpStatusCode.ServiceUnavailable));
  }

  private sealed class Rig
  {
    public readonly TestSchedulerProvider Schedulers = new();
    public readonly Mock<INativeRuntimeService> Runtime = new();
    public readonly TimelineSession Session = new();
    public readonly ForwardingCometMessenger Messenger = new();
    public readonly TimelineService Timeline;
    public readonly CometTabViewModel Vm;
    public readonly TimelineTabViewModel Tl;

    public Rig()
    {
      var translation = new Mock<ITranslationService>();
      translation.Setup(t => t.CultureChanged).Returns(Observable.Empty<System.Globalization.CultureInfo>());
      translation.Setup(t => t.GetString(It.IsAny<string>())).Returns<string>(k => k);

      var cometSessions = new Mock<ITabStateService<CometSession>>();
      cometSessions.Setup(s => s.ActiveSessionIds).Returns(new ObservableCollection<SessionId> { new SessionId(typeof(CometSession), 1) });
      cometSessions.Setup(x => x.ObserveSession(It.IsAny<SessionId>())).Returns(Observable.Return(new CometSession()));
      cometSessions.Setup(s => s.ObserveSessionList()).Returns(new BehaviorSubject<IReadOnlyList<SessionId>>(new List<SessionId>()));

      // as TabStateService: every update is published to the session's observers
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
      var jpl = new HorizonJplService(new ConsoleService(new Mock<IUiThreadDispatcher>().Object), breadcrumbs, storage.Object) { _httpClient = new HttpClient(new OfflineHandler()) };
      Timeline = new TimelineService(Runtime.Object, Schedulers, config, breadcrumbs);

      Vm = new CometTabViewModel(translation.Object, Schedulers, cometSessions.Object, jpl, config, Timeline,
        breadcrumbs, storage.Object, Runtime.Object, new Mock<ITabStateService<ModelSession>>().Object,
        Messenger, new Mock<ITabFactory>().Object, timelineSessions.Object);
      Tl = new TimelineTabViewModel(translation.Object, Schedulers, timelineSessions.Object, cometSessions.Object,
        Timeline, Runtime.Object, Messenger);
      Tl.CurrentSession = Session;
      Flush();
    }

    /// <summary>Delivers the main-thread work (proposals, recommit requests). A proposal hops the
    /// main-thread scheduler twice (TimelineService's ObserveOn, then the tab's), one tick each;
    /// a few ticks stay far below the 33 ms play poll.</summary>
    public void Flush()
    {
      for (int i = 0; i < 4; i++) Schedulers.MainThread.AdvanceBy(1);
    }

    /// <summary>
    /// What a successful Download &amp; Commit does after the native commit: the committed range
    /// (the proposal it applied) into the range entry, then the message to every tab.
    /// </summary>
    public void CompleteCommit()
    {
      var range = Vm.ProposedRangeForCommit;
      Vm.IsAlmanacCommitted = true;
      Vm.Range.OnCommitted(range);
      Messenger.Send(new CometCommittedMessage(range));
      Flush();
    }

    public void SelectCometAndRecord()
    {
      Vm.FilteredSearchResults = new ObservableCollection<CometSearchResult>
      {
        new() { Name = "67P/Churyumov-Gerasimenko", PrimaryDesignation = "67P" },
      };
      Vm.SelectedComet = Vm.FilteredSearchResults[0];
      Vm.SelectedSpkRecord = new SpkRecordItem { RecordId = "90000703", Name = "67P" };
      Vm.ReferenceOrbitIndex = 0;
    }
  }

  private static TimeRange Range(DateTimeOffset start, DateTimeOffset end)
  {
    var a = TimeUtils.ToTaiParts(start);
    var b = TimeUtils.ToTaiParts(end);
    return new TimeRange(a.centuries, a.nanoseconds, b.centuries, b.nanoseconds);
  }

  private static readonly DateTimeOffset Dec12 = new(2021, 12, 12, 0, 0, 0, TimeSpan.Zero);
  private static readonly DateTimeOffset Dec17 = new(2021, 12, 17, 15, 48, 0, TimeSpan.Zero);
  private static readonly DateTimeOffset Dec28 = new(2021, 12, 28, 0, 0, 0, TimeSpan.Zero);

  // ── D5: the range lives in the Comet tab ─────────────────────────────────

  [Fact]
  public void EditingTheRangeInTheCometTab_ProposesIt_AndTheCommitUsesIt()
  {
    var rig = new Rig();
    // the default proposal is loaded into the boxes and proposed
    Assert.Equal(TimelineSession.DefaultProposedStart, rig.Vm.Range.StartEpoch);
    Assert.True(rig.Vm.HasProposedTimeline);

    rig.Vm.Range.StartEpoch = "2021-12-12T00:00:00";
    rig.Vm.Range.EndEpoch = "2021-12-28T00:00:00";
    rig.Flush();

    Assert.Equal("2021-12-12T00:00:00", rig.Session.ProposedStartEpoch);
    Assert.Equal("2021-12-28T00:00:00", rig.Session.ProposedEndEpoch);
    Assert.Equal("2021-12-12 00:00", rig.Vm.ProposedStartEpoch); // the SPK download window
    Assert.Equal("2021-12-28 00:00", rig.Vm.ProposedEndEpoch);
    Assert.Equal(Range(Dec12, Dec28), rig.Vm.ProposedRangeForCommit); // RequestEpochRange / CommitCometAsync
    Assert.False(rig.Vm.Range.HasCommittedState);

    rig.SelectCometAndRecord();
    Assert.True(rig.Vm.DownloadAndCommitCommand.CanExecute(null));
    rig.CompleteCommit();

    Assert.Equal("2021-12-12T00:00:00", rig.Session.CommittedStartEpoch);
    Assert.Equal("2021-12-28T00:00:00", rig.Session.CommittedEndEpoch);
    Assert.True(rig.Tl.HasCommittedState);
    Assert.Equal("2021-12-12T00:00:00 to 2021-12-28T00:00:00", rig.Tl.DisplayCommittedRange);
    Assert.False(rig.Vm.Range.IsRangeLocked);
  }

  [Fact]
  public void AfterACommit_AnEditShowsTheLockText_AndRecommitAppliesIt()
  {
    var rig = new Rig();
    rig.Vm.Range.StartEpoch = "2021-12-12T00:00:00";
    rig.Vm.Range.EndEpoch = "2021-12-28T00:00:00";
    rig.Flush();
    rig.SelectCometAndRecord();
    rig.CompleteCommit();
    rig.Vm.CommittedCometName = "67P/Churyumov-Gerasimenko";
    rig.Vm.CommittedSpkRecordId = "90000703";
    rig.Vm.HasTimelineChangedAfterCommit = false;
    Assert.False(rig.Vm.DownloadAndCommitCommand.CanExecute(null)); // nothing new to commit

    int requests = 0;
    using var sub = rig.Timeline.RecommitRequests.Subscribe(_ => requests++);
    rig.Vm.Range.StartEpoch = "2021-12-17T15:48:00";
    rig.Flush();

    // the lock text is shown here, the engine keeps the committed range
    Assert.True(rig.Vm.Range.IsRangeLocked);
    Assert.False(string.IsNullOrEmpty(RangeEntryViewModel.RangeLockedText));
    Assert.Equal("2021-12-12T00:00:00", rig.Session.CommittedStartEpoch);
    Assert.Equal("2021-12-12T00:00:00", rig.Tl.StartEpoch);
    Assert.True(rig.Tl.HasPendingRangeEdit);
    rig.Runtime.Verify(r => r.SetEpochRange(It.IsAny<short>(), It.IsAny<ulong>(), It.IsAny<short>(), It.IsAny<ulong>()), Times.Never);

    // Recommit: the Comet tab's Download & Commit, now enabled by the lock, with the edited range
    Assert.True(rig.Vm.DownloadAndCommitCommand.CanExecute(null));
    rig.Vm.Range.RecommitWithRangeCommand.Execute(null);
    rig.Flush();
    Assert.Equal(1, requests);
    Assert.Equal(Range(Dec17, Dec28), rig.Vm.ProposedRangeForCommit);

    rig.CompleteCommit();
    Assert.Equal("2021-12-17T15:48:00", rig.Session.CommittedStartEpoch);
    Assert.False(rig.Vm.Range.IsRangeLocked);
    Assert.Equal("2021-12-17T15:48:00", rig.Tl.StartEpoch);
    Assert.False(rig.Tl.HasPendingRangeEdit);
  }

  [Fact]
  public void TheTimeline_ExposesTheRangeReadOnly_AndItsBoxesEqualTheCommittedRange()
  {
    foreach (var name in new[] { nameof(TimelineTabViewModel.StartEpoch), nameof(TimelineTabViewModel.EndEpoch) })
    {
      var p = typeof(TimelineTabViewModel).GetProperty(name)!;
      Assert.False(p.SetMethod?.IsPublic ?? false, $"{name} must be read-only");
    }
    // no range entry left on the Timeline
    foreach (var gone in new[] { "RestoreCommand", "RecommitWithRangeCommand", "ApplyModelRangeCommand", "DustLeadDays", "IsRangeLocked", "ModelRotations", "ModelEndEpoch" })
      Assert.Null(typeof(TimelineTabViewModel).GetProperty(gone));

    var rig = new Rig();
    // before a commit: the proposal, as typed in the Comet tab
    rig.Vm.Range.StartEpoch = "2021-12-12T00:00:00";
    rig.Vm.Range.EndEpoch = "2021-12-28T00:00:00";
    Assert.False(rig.Tl.HasCommittedState);
    Assert.Equal("2021-12-12T00:00:00", rig.Tl.StartEpoch);
    Assert.Equal("2021-12-28T00:00:00", rig.Tl.EndEpoch);

    rig.Flush();
    rig.SelectCometAndRecord();
    rig.CompleteCommit();
    Assert.Equal(rig.Session.CommittedStartEpoch, rig.Tl.StartEpoch);
    Assert.Equal(rig.Session.CommittedEndEpoch, rig.Tl.EndEpoch);

    // a decommit from the Comet tab: the committed range flows back as the proposal
    rig.Vm.IsAlmanacCommitted = true;
    rig.Vm.DecommitCometCommand.Execute(null);
    Assert.False(rig.Tl.HasCommittedState);
    Assert.Equal("2021-12-12T00:00:00", rig.Tl.StartEpoch);
    Assert.Equal("2021-12-12T00:00:00", rig.Vm.Range.StartEpoch);
  }

  [Fact]
  public void ModelRange_SetsStartToEndMinusNRotationPeriods()
  {
    var rig = new Rig();
    rig.Vm.RotRateDegDay = 696.0; // 67P: P = 360 · 24 / 696 h ≈ 12.41 h
    double periodHours = 360.0 * 24.0 / 696.0;
    rig.Vm.Range.ModelRotations = 3;
    rig.Vm.Range.ModelEndEpoch = "2021-12-27T00:14:47";
    rig.Vm.Range.ApplyModelRangeCommand.Execute(null);

    var end = new DateTimeOffset(2021, 12, 27, 0, 14, 47, TimeSpan.Zero);
    Assert.Equal("2021-12-27T00:14:47", rig.Vm.Range.EndEpoch);
    Assert.True(TimeUtils.TryParseIso8601(rig.Vm.Range.StartEpoch, out var start));
    Assert.InRange((end - start).TotalHours - 3 * periodHours, -1.0 / 3600, 1.0 / 3600);
    Assert.False(rig.Vm.Range.HasError);
    Assert.Equal(rig.Vm.Range.StartEpoch, rig.Session.ProposedStartEpoch);

    // the FITS row's "Model range": the same, through TimelineService.RequestModelRange
    rig.Vm.Range.ModelRotations = 1;
    var image = new DateTimeOffset(2021, 12, 20, 6, 0, 0, TimeSpan.Zero);
    rig.Timeline.RequestModelRange(image);
    rig.Flush();
    Assert.Equal("2021-12-20T06:00:00", rig.Vm.Range.ModelEndEpoch);
    Assert.Equal("2021-12-20T06:00:00", rig.Vm.Range.EndEpoch);
    Assert.True(TimeUtils.TryParseIso8601(rig.Vm.Range.StartEpoch, out var s1));
    // one 12.41 h rotation is under a day: the minimum range applies
    Assert.Equal(image - TimeSpan.FromDays(RangeEntryViewModel.MinRangeDays), s1);

    // no rotation: refused, the boxes kept
    rig.Vm.RotRateDegDay = 0;
    string before = rig.Vm.Range.StartEpoch;
    rig.Vm.Range.ApplyModelRangeCommand.Execute(null);
    Assert.True(rig.Vm.Range.HasError);
    Assert.Contains("rotation period", rig.Vm.Range.ErrorMessage);
    Assert.Equal(before, rig.Vm.Range.StartEpoch);
  }

  /// <summary>
  /// P4: the reviewer's FITS frame (DATE-OBS 2021-12-27T00:14:37.345, 20 s exposure) through its
  /// row's "Model range" button, wired as in the app (the billboards' request goes to
  /// TimelineService.RequestModelRange): the Comet tab's "rotations ending at" box shows the
  /// mid-exposure in the box format, and the range is N rotations ending there.
  /// </summary>
  [Fact]
  public void FitsModelRange_FillsTheCometTabModelEndBox_WithTheImageMidExposure()
  {
    var rig = new Rig();
    const double rateDegDay = 696.2127316680096; // the reviewer's save: P ≈ 12.41 h
    rig.Vm.RotRateDegDay = rateDegDay;
    rig.Vm.Range.ModelRotations = 2;
    var fits = new FitsBillboardTests.Rig(onModelRange: rig.Timeline.RequestModelRange);
    var row = fits.Create("/data/lt_67p.fits");

    Assert.True(row.Fits!.RequestModelRangeCommand.CanExecute(null));
    row.Fits.RequestModelRangeCommand.Execute(null);
    fits.Fake.Drain();
    rig.Flush();

    Assert.Equal("2021-12-27T00:14:47", rig.Vm.Range.ModelEndEpoch); // the box: no fraction, no Z
    Assert.Equal("2021-12-27T00:14:47", rig.Vm.Range.EndEpoch);
    Assert.True(TimeUtils.TryParseIso8601(rig.Vm.Range.StartEpoch, out var start));
    var end = new DateTimeOffset(2021, 12, 27, 0, 14, 47, TimeSpan.Zero);
    double periodHours = 360.0 * 24.0 / rateDegDay;
    Assert.InRange((end - start).TotalHours - 2 * periodHours, -1.0 / 3600, 1.0 / 3600);
    Assert.False(rig.Vm.Range.HasError);
    Assert.Equal(rig.Vm.Range.EndEpoch, rig.Session.ProposedEndEpoch);
  }

  private static DateTimeOffset Day(string yyyyMmDd) =>
    DateTimeOffset.ParseExact(yyyyMmDd, "yyyy-MM-dd", System.Globalization.CultureInfo.InvariantCulture,
      System.Globalization.DateTimeStyles.AssumeUniversal);

  /// <summary>The SPK request (START_TIME, STOP_TIME) the Comet tab's Download &amp; Commit makes for a range.</summary>
  private static (DateTimeOffset Start, DateTimeOffset Stop) SpkRequest(TimeRange range)
  {
    var (a, b) = CometTabViewModel.SpkDownloadDays(range);
    var (start, stop) = HorizonJplService.SpkRequestDays(a, b);
    return (Day(start), Day(stop));
  }

  /// <summary>
  /// docx 19 ("it picks an end date four years away"): the ±5 years around today only list the SPK
  /// records; the SPK download is the range the commit applies, widened by the margin, so it
  /// always holds the range, for ranges near midnight and ranges more than five years ago.
  /// </summary>
  [Theory]
  [InlineData("2021-12-12T00:00:00Z", "2021-12-28T00:00:00Z")]       // the reviewer's committed range
  [InlineData("2021-12-26T23:59:59.999Z", "2021-12-27T23:59:59.999Z")] // a day ending just before midnight
  [InlineData("2014-08-01T00:00:00Z", "2014-11-12T12:00:00Z")]       // twelve years before today
  [InlineData(TimelineSession.DefaultProposedStart, TimelineSession.DefaultProposedEnd)]
  public void SpkDownloadWindow_HoldsTheRange(string startText, string endText)
  {
    Assert.True(TimeUtils.TryParseIso8601(startText, out var start));
    Assert.True(TimeUtils.TryParseIso8601(endText, out var end));
    var (s, e) = SpkRequest(Range(start, end));
    Assert.True(s <= start - TimeSpan.FromDays(HorizonJplService.SpkMarginDays - 1), $"SPK start {s:O} after {start:O}");
    Assert.True(e >= end + TimeSpan.FromDays(HorizonJplService.SpkMarginDays - 1), $"SPK stop {e:O} before {end:O}");
  }

  /// <summary>
  /// docx 19 through the tab: the proposal the Download &amp; Commit applies (typed, then the FITS
  /// model range) and the range it commits all lie inside the SPK it downloads.
  /// </summary>
  [Fact]
  public void SpkDownloadWindow_HoldsTheProposedAndTheCommittedRange()
  {
    var rig = new Rig();
    rig.Vm.Range.StartEpoch = "2021-12-12T00:00:00";
    rig.Vm.Range.EndEpoch = "2021-12-28T00:00:00";
    rig.Flush();
    var (s, e) = SpkRequest(rig.Vm.ProposedRangeForCommit!);
    Assert.True(s <= Dec12 && e >= Dec28, $"SPK {s:O} to {e:O}");

    rig.SelectCometAndRecord();
    rig.CompleteCommit();
    Assert.True(TimeUtils.TryParseIso8601(rig.Session.CommittedStartEpoch, out var cs));
    Assert.True(TimeUtils.TryParseIso8601(rig.Session.CommittedEndEpoch, out var ce));
    Assert.True(s <= cs && e >= ce, $"committed {cs:O} to {ce:O} outside SPK {s:O} to {e:O}");

    // a model range ending at the image epoch, proposed while committed (the Recommit path)
    rig.Vm.RotRateDegDay = 696.0;
    rig.Timeline.RequestModelRange(new DateTimeOffset(2021, 12, 27, 0, 14, 47, TimeSpan.Zero));
    rig.Flush();
    var proposal = rig.Vm.ProposedRangeForCommit!;
    var (s2, e2) = SpkRequest(proposal);
    Assert.True(s2 <= TimeUtils.FromTaiParts(proposal.StartCenturies, proposal.StartNs));
    Assert.True(e2 >= TimeUtils.FromTaiParts(proposal.EndCenturies, proposal.EndNs));
    Assert.Equal(new DateTimeOffset(2021, 12, 27, 0, 14, 47, TimeSpan.Zero), TimeUtils.FromTaiParts(proposal.EndCenturies, proposal.EndNs));
  }

  [Fact]
  public void DustLeadTime_IsClampedAndKeptInTheTimelineSession()
  {
    var rig = new Rig();
    rig.Vm.Range.DustLeadDays = 7.5;
    Assert.Equal(7.5, rig.Timeline.DustLeadDays);
    Assert.Equal(7.5, rig.Session.DustLeadDays);
    rig.Vm.Range.DustLeadDays = 99;
    Assert.Equal(TimelineService.MaxDustLeadDays, rig.Vm.Range.DustLeadDays);
    Assert.Equal(TimelineService.MaxDustLeadDays, rig.Timeline.DustLeadDays);
  }

  // ── D6: the guided step strip ────────────────────────────────────────────

  [Theory]
  //         results comet  record valid  committed locked  step
  [InlineData(false, false, false, true,  false, false, 1)]
  [InlineData(true,  false, false, true,  false, false, 2)]
  [InlineData(true,  true,  false, true,  false, false, 2)]
  [InlineData(true,  true,  true,  false, false, false, 3)]
  [InlineData(true,  true,  true,  true,  false, false, 4)]
  [InlineData(true,  true,  true,  true,  true,  false, 5)]
  // a reopened save: committed without search results or a selection
  [InlineData(false, false, false, true,  true,  false, 5)]
  // edited after the commit: recommit (4), or fix the range first (3)
  [InlineData(true,  true,  true,  true,  true,  true,  4)]
  [InlineData(true,  true,  true,  false, true,  true,  3)]
  public void GuidedStep_ForEachState(bool results, bool comet, bool record, bool valid, bool committed, bool locked, int step)
  {
    Assert.Equal(step, CometTabViewModel.GuidedStepFor(results, comet, record, valid, committed, locked));
  }

  [Fact]
  public void GuidedStep_FollowsTheTab_DoneStepsTicked_CurrentHighlighted()
  {
    var rig = new Rig();
    var vm = rig.Vm;
    void AssertStep(int n)
    {
      Assert.Equal(n, vm.CurrentGuidedStep);
      Assert.Equal(new[] { n > 1, n > 2, n > 3, n > 4 }, new[] { vm.IsStep1Done, vm.IsStep2Done, vm.IsStep3Done, vm.IsStep4Done });
      Assert.Equal(new[] { n == 1, n == 2, n == 3, n == 4 }, new[] { vm.IsStep1Current, vm.IsStep2Current, vm.IsStep3Current, vm.IsStep4Current });
      Assert.Equal(n <= 4 ? $"Tabs_Comet_Step{n}Help" : "Tabs_Comet_StepsDoneHelp", vm.CurrentStepHelp);
    }

    AssertStep(1); // nothing searched
    vm.FilteredSearchResults = new ObservableCollection<CometSearchResult>
    {
      new() { Name = "67P/Churyumov-Gerasimenko", PrimaryDesignation = "67P" },
    };
    AssertStep(2); // results: pick the comet and its record
    vm.SelectedComet = vm.FilteredSearchResults[0];
    AssertStep(2);
    vm.SelectedSpkRecord = new SpkRecordItem { RecordId = "90000703", Name = "67P" };
    AssertStep(4); // the default range is valid: commit

    vm.Range.EndEpoch = "2025-10-01T06:00:00Z"; // less than a day
    AssertStep(3);
    vm.Range.EndEpoch = "2025-11-11T00:00:00Z";
    AssertStep(4);

    rig.Flush();
    vm.ReferenceOrbitIndex = 0;
    rig.CompleteCommit();
    AssertStep(5);

    vm.Range.StartEpoch = "2025-10-05T00:00:00Z"; // edited after the commit: recommit
    AssertStep(4);
    vm.Range.RestoreCommand.Execute(null);
    AssertStep(5);
  }
}
