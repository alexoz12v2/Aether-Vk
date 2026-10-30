using System;
using AetherVk.Logic.Messages;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using AetherVk.Logic.Tests.Mocks;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The committed range is what runs (reviewer report 2026-10-29, "the timeline picks other
/// dates"): the reviewer's save was committed 2021-12-12 → 28 with "2021-12-17T15:48:00" left in
/// the start box, and the commit copied the boxes into the committed range while the engine ran
/// the 12-12 one. An edit after a commit is locked, said so, and can be recommitted.
///
/// <para>Since reviewer feedback item 7 the range is entered in the Comet tab
/// (<see cref="RangeEntryViewModel"/>) and the Timeline tab shows it read-only: these are the
/// former Timeline range-lock tests, run on the range entry, with the Timeline tab beside it over
/// the same session.</para>
/// </summary>
public class TimelineRangeLockTests
{
  private sealed class Rig
  {
    public readonly Mock<INativeRuntimeService> Runtime = new();
    public readonly TimelineSession Session = new();
    public readonly TimelineService Timeline;
    public readonly RangeEntryViewModel Entry;
    public readonly TimelineTabViewModel Tl;
    public readonly TestSchedulerProvider Schedulers = new();

    public Rig()
    {
      var translation = new Mock<ITranslationService>();
      translation.Setup(t => t.CultureChanged).Returns(System.Reactive.Linq.Observable.Empty<System.Globalization.CultureInfo>());
      var sessions = new Mock<ITabStateService<TimelineSession>>();
      sessions.Setup(s => s.ActiveSessionIds).Returns(new System.Collections.ObjectModel.ObservableCollection<SessionId> { new SessionId(typeof(TimelineSession), 1) });
      sessions.Setup(s => s.ObserveSessionList()).Returns(new System.Reactive.Subjects.BehaviorSubject<System.Collections.Generic.IReadOnlyList<SessionId>>(new System.Collections.Generic.List<SessionId>()));
      var comets = new Mock<ITabStateService<CometSession>>();
      comets.Setup(s => s.ActiveSessionIds).Returns(new System.Collections.ObjectModel.ObservableCollection<SessionId> { new SessionId(typeof(CometSession), 1) });
      comets.Setup(x => x.ObserveSession(It.IsAny<SessionId>())).Returns(System.Reactive.Linq.Observable.Return(new CometSession()));
      comets.Setup(s => s.ObserveSessionList()).Returns(new System.Reactive.Subjects.BehaviorSubject<System.Collections.Generic.IReadOnlyList<SessionId>>(new System.Collections.Generic.List<SessionId>()));
      Timeline = new TimelineService(Runtime.Object, Schedulers, new CometConfigService(Runtime.Object, Schedulers), new BreadcrumbService());

      sessions.Setup(x => x.GetSession(It.IsAny<SessionId>())).Returns(Session);
      // as TabStateService: every update is published to the session's observers
      var stream = new System.Reactive.Subjects.BehaviorSubject<TimelineSession>(Session);
      sessions.Setup(x => x.ObserveSession(It.IsAny<SessionId>())).Returns(stream);
      sessions.Setup(x => x.UpdateSession(It.IsAny<SessionId>(), It.IsAny<Action<TimelineSession>>()))
        .Callback<SessionId, Action<TimelineSession>>((id, action) => { action(Session); stream.OnNext(Session); });
      Runtime.Setup(r => r.GetSimulationClock(out It.Ref<short>.IsAny, out It.Ref<ulong>.IsAny)).Returns(false);

      Entry = new RangeEntryViewModel(sessions.Object, Timeline, null, () => 0);
      Tl = new TimelineTabViewModel(translation.Object, Schedulers, sessions.Object, comets.Object,
        Timeline, Runtime.Object, new Mock<ICometMessenger>().Object);
      Tl.CurrentSession = Session;
    }

    /// <summary>A commit with this range, as the Comet tab and the messenger deliver it.</summary>
    public void Commit(string start, string end)
    {
      var a = TimeUtils.ToTaiParts(DateTimeOffset.Parse(start + "Z", System.Globalization.CultureInfo.InvariantCulture));
      var b = TimeUtils.ToTaiParts(DateTimeOffset.Parse(end + "Z", System.Globalization.CultureInfo.InvariantCulture));
      var range = new TimeRange(a.centuries, a.nanoseconds, b.centuries, b.nanoseconds);
      Entry.OnCommitted(range);
      Tl.Receive(new CometCommittedMessage(range));
    }
  }

  [Fact]
  public void ACommit_ShowsTheRangeItWasCommittedWith_NotTheBoxes()
  {
    var rig = new Rig();
    // the reviewer's save: a later proposal in the start box
    rig.Entry.StartEpoch = "2021-12-17T15:48:00";
    rig.Entry.EndEpoch = "2021-12-28T00:00:00";

    rig.Commit("2021-12-12T00:00:00", "2021-12-28T00:00:00");

    Assert.Equal("2021-12-12T00:00:00Z", rig.Session.CommittedStartEpoch);
    Assert.Equal("2021-12-28T00:00:00", rig.Session.CommittedEndEpoch); // same instant: the typed text stays
    Assert.Equal(rig.Session.CommittedStartEpoch, rig.Entry.StartEpoch);
    Assert.Equal(rig.Session.CommittedEndEpoch, rig.Entry.EndEpoch);
    Assert.False(rig.Entry.IsProposedDifferentFromCommitted);
    Assert.False(rig.Entry.IsRangeLocked);
    // the Timeline shows the same committed range
    Assert.Equal(rig.Session.CommittedStartEpoch, rig.Tl.StartEpoch);
    Assert.Equal(rig.Session.CommittedEndEpoch, rig.Tl.EndEpoch);
    Assert.False(rig.Tl.HasPendingRangeEdit);
  }

  [Fact]
  public void AnEditAfterTheCommit_IsLocked_AndLeavesTheCommittedRange()
  {
    var rig = new Rig();
    rig.Entry.StartEpoch = "2021-12-12T00:00:00";
    rig.Entry.EndEpoch = "2021-12-28T00:00:00";
    rig.Commit("2021-12-12T00:00:00", "2021-12-28T00:00:00");

    rig.Entry.StartEpoch = "2021-12-17T15:48:00";

    Assert.True(rig.Entry.IsRangeLocked);
    Assert.Equal("2021-12-12T00:00:00", rig.Session.CommittedStartEpoch);
    Assert.Equal("2021-12-12T00:00:00 to 2021-12-28T00:00:00", rig.Entry.DisplayCommittedRange);
    Assert.Equal("2021-12-12T00:00:00 to 2021-12-28T00:00:00", rig.Tl.DisplayCommittedRange);
    rig.Runtime.Verify(r => r.SetEpochRange(It.IsAny<short>(), It.IsAny<ulong>(), It.IsAny<short>(), It.IsAny<ulong>()), Times.Never);
    // the Timeline keeps showing what runs, and says an edit waits
    Assert.Equal("2021-12-12T00:00:00", rig.Tl.StartEpoch);
    Assert.True(rig.Tl.HasPendingRangeEdit);

    // Restore: the committed range back in the boxes, and the lock clears
    rig.Entry.RestoreCommand.Execute(null);
    Assert.Equal("2021-12-12T00:00:00", rig.Entry.StartEpoch);
    Assert.False(rig.Entry.IsRangeLocked);
  }

  [Fact]
  public void Recommit_AsksTheCometTab_OnlyForALockedValidRange()
  {
    var rig = new Rig();
    int requests = 0;
    using var sub = rig.Timeline.RecommitRequests.Subscribe(_ => requests++);
    rig.Entry.StartEpoch = "2021-12-12T00:00:00";
    rig.Entry.EndEpoch = "2021-12-28T00:00:00";
    rig.Commit("2021-12-12T00:00:00", "2021-12-28T00:00:00");

    rig.Entry.RecommitWithRangeCommand.Execute(null); // nothing to recommit
    rig.Entry.StartEpoch = "2021-12-27T12:00:00";     // less than a day: invalid
    rig.Entry.RecommitWithRangeCommand.Execute(null);
    rig.Schedulers.MainThread.AdvanceBy(1);
    Assert.Equal(0, requests);

    rig.Entry.StartEpoch = "2021-12-17T15:48:00";
    rig.Entry.RecommitWithRangeCommand.Execute(null);
    rig.Schedulers.MainThread.AdvanceBy(1);
    Assert.Equal(1, requests);
  }

  [Fact]
  public void ADecommit_UnlocksAndKeepsTheCommittedRangeInTheBoxes()
  {
    var rig = new Rig();
    rig.Entry.StartEpoch = "2021-12-12T00:00:00";
    rig.Entry.EndEpoch = "2021-12-28T00:00:00";
    rig.Commit("2021-12-12T00:00:00", "2021-12-28T00:00:00");
    rig.Entry.StartEpoch = "2021-12-17T15:48:00";

    rig.Entry.OnDecommitted();
    rig.Tl.Receive(new CometDecommittedMessage());

    Assert.False(rig.Entry.IsRangeLocked);
    Assert.False(rig.Entry.HasCommittedState);
    Assert.False(rig.Tl.HasCommittedState);
    Assert.Equal("2021-12-12T00:00:00", rig.Entry.StartEpoch);
    Assert.Equal("2021-12-12T00:00:00", rig.Session.ProposedStartEpoch);
    Assert.Equal("2021-12-12T00:00:00", rig.Tl.StartEpoch);
  }

  /// <summary>The Comet tab and the Timeline tab both apply a commit / decommit message: twice, in
  /// either order, gives the same session.</summary>
  [Fact]
  public void CommitAndDecommit_AreIdempotent_InEitherOrder()
  {
    var rig = new Rig();
    rig.Entry.StartEpoch = "2021-12-17T15:48:00";
    rig.Entry.EndEpoch = "2021-12-28T00:00:00";
    var a = TimeUtils.ToTaiParts(new DateTimeOffset(2021, 12, 12, 0, 0, 0, TimeSpan.Zero));
    var b = TimeUtils.ToTaiParts(new DateTimeOffset(2021, 12, 28, 0, 0, 0, TimeSpan.Zero));
    var range = new TimeRange(a.centuries, a.nanoseconds, b.centuries, b.nanoseconds);

    rig.Tl.Receive(new CometCommittedMessage(range)); // the Timeline first this time
    rig.Entry.OnCommitted(range);
    rig.Tl.Receive(new CometCommittedMessage(range));
    Assert.Equal("2021-12-12T00:00:00Z", rig.Session.CommittedStartEpoch);
    Assert.Equal("2021-12-28T00:00:00", rig.Session.CommittedEndEpoch);
    Assert.Equal(rig.Session.CommittedStartEpoch, rig.Entry.StartEpoch);
    Assert.False(rig.Entry.IsRangeLocked);

    rig.Tl.Receive(new CometDecommittedMessage());
    rig.Entry.OnDecommitted();
    rig.Entry.OnDecommitted();
    Assert.Equal(string.Empty, rig.Session.CommittedStartEpoch);
    Assert.Equal("2021-12-12T00:00:00Z", rig.Session.ProposedStartEpoch);
    Assert.Equal("2021-12-12T00:00:00Z", rig.Entry.StartEpoch);
  }
}
