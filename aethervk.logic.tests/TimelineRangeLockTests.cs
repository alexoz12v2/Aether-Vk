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
/// The committed range is what the Timeline shows (reviewer report 2026-10-29, "the timeline picks
/// other dates"): his save was committed 2021-12-12 → 28 with "2021-12-17T15:48:00" left in the
/// start box, and the commit copied the boxes into the committed range while the engine ran the
/// 12-12 one. An edit after a commit is locked, said so, and can be recommitted.
/// </summary>
public class TimelineRangeLockTests
{
  private sealed class Rig
  {
    public readonly Mock<INativeRuntimeService> Runtime = new();
    public readonly TimelineSession Session = new();
    public readonly TimelineService Timeline;
    public readonly TimelineTabViewModel Vm;
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
      sessions.Setup(x => x.ObserveSession(It.IsAny<SessionId>())).Returns(System.Reactive.Linq.Observable.Return(Session));
      sessions.Setup(x => x.UpdateSession(It.IsAny<SessionId>(), It.IsAny<Action<TimelineSession>>()))
        .Callback<SessionId, Action<TimelineSession>>((id, action) => action(Session));
      Runtime.Setup(r => r.GetSimulationClock(out It.Ref<short>.IsAny, out It.Ref<ulong>.IsAny)).Returns(false);

      Vm = new TimelineTabViewModel(translation.Object, Schedulers, sessions.Object, comets.Object,
        Timeline, Runtime.Object, new Mock<ICometMessenger>().Object);
      Vm.CurrentSession = Session;
    }

    public void Commit(string start, string end)
    {
      var a = TimeUtils.ToTaiParts(DateTimeOffset.Parse(start + "Z", System.Globalization.CultureInfo.InvariantCulture));
      var b = TimeUtils.ToTaiParts(DateTimeOffset.Parse(end + "Z", System.Globalization.CultureInfo.InvariantCulture));
      Vm.Receive(new CometCommittedMessage(new TimeRange(a.centuries, a.nanoseconds, b.centuries, b.nanoseconds)));
    }
  }

  [Fact]
  public void ACommit_ShowsTheRangeItWasCommittedWith_NotTheBoxes()
  {
    var rig = new Rig();
    // the reviewer's save: a later proposal in the start box
    rig.Vm.StartEpoch = "2021-12-17T15:48:00";
    rig.Vm.EndEpoch = "2021-12-28T00:00:00";

    rig.Commit("2021-12-12T00:00:00", "2021-12-28T00:00:00");

    Assert.Equal("2021-12-12T00:00:00Z", rig.Session.CommittedStartEpoch);
    Assert.Equal("2021-12-28T00:00:00", rig.Session.CommittedEndEpoch); // same instant: the typed text stays
    Assert.Equal(rig.Session.CommittedStartEpoch, rig.Vm.StartEpoch);
    Assert.Equal(rig.Session.CommittedEndEpoch, rig.Vm.EndEpoch);
    Assert.False(rig.Vm.IsProposedDifferentFromCommitted);
    Assert.False(rig.Vm.IsRangeLocked);
  }

  [Fact]
  public void AnEditAfterTheCommit_IsLocked_AndLeavesTheCommittedRange()
  {
    var rig = new Rig();
    rig.Vm.StartEpoch = "2021-12-12T00:00:00";
    rig.Vm.EndEpoch = "2021-12-28T00:00:00";
    rig.Commit("2021-12-12T00:00:00", "2021-12-28T00:00:00");

    rig.Vm.StartEpoch = "2021-12-17T15:48:00";

    Assert.True(rig.Vm.IsRangeLocked);
    Assert.Equal("2021-12-12T00:00:00", rig.Session.CommittedStartEpoch);
    Assert.Equal("2021-12-12T00:00:00 to 2021-12-28T00:00:00", rig.Vm.DisplayCommittedRange);
    rig.Runtime.Verify(r => r.SetEpochRange(It.IsAny<short>(), It.IsAny<ulong>(), It.IsAny<short>(), It.IsAny<ulong>()), Times.Never);

    // Restore: the committed range back in the boxes, and the lock clears
    rig.Vm.RestoreCommand.Execute(null);
    Assert.Equal("2021-12-12T00:00:00", rig.Vm.StartEpoch);
    Assert.False(rig.Vm.IsRangeLocked);
  }

  [Fact]
  public void Recommit_AsksTheCometTab_OnlyForALockedValidRange()
  {
    var rig = new Rig();
    int requests = 0;
    using var sub = rig.Timeline.RecommitRequests.Subscribe(_ => requests++);
    rig.Vm.StartEpoch = "2021-12-12T00:00:00";
    rig.Vm.EndEpoch = "2021-12-28T00:00:00";
    rig.Commit("2021-12-12T00:00:00", "2021-12-28T00:00:00");

    rig.Vm.RecommitWithRangeCommand.Execute(null); // nothing to recommit
    rig.Vm.StartEpoch = "2021-12-27T12:00:00";     // less than a day: invalid
    rig.Vm.RecommitWithRangeCommand.Execute(null);
    rig.Schedulers.MainThread.AdvanceBy(1);
    Assert.Equal(0, requests);

    rig.Vm.StartEpoch = "2021-12-17T15:48:00";
    rig.Vm.RecommitWithRangeCommand.Execute(null);
    rig.Schedulers.MainThread.AdvanceBy(1);
    Assert.Equal(1, requests);
  }

  [Fact]
  public void ADecommit_UnlocksAndKeepsTheCommittedRangeInTheBoxes()
  {
    var rig = new Rig();
    rig.Vm.StartEpoch = "2021-12-12T00:00:00";
    rig.Vm.EndEpoch = "2021-12-28T00:00:00";
    rig.Commit("2021-12-12T00:00:00", "2021-12-28T00:00:00");
    rig.Vm.StartEpoch = "2021-12-17T15:48:00";

    rig.Vm.Receive(new CometDecommittedMessage());

    Assert.False(rig.Vm.IsRangeLocked);
    Assert.False(rig.Vm.HasCommittedState);
    Assert.Equal("2021-12-12T00:00:00", rig.Vm.StartEpoch);
  }
}
