using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using AetherVk.Logic.Services;
using AetherVk.Logic.Tests.Mocks;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// Savefile open replaces the singleton sessions' values and raises <c>Restored</c>: the live
/// Timeline and Comet tabs re-read them; a disposed tab no longer listens.
/// </summary>
public class SessionRestoredTabTests
{
  private static Mock<ITabStateService<T>> SessionService<T>(T session) where T : class, ITabSession, new()
  {
    var m = new Mock<ITabStateService<T>>();
    m.Setup(s => s.ActiveSessionIds).Returns(new ObservableCollection<SessionId> { new(typeof(T), 1) });
    m.Setup(s => s.ObserveSession(It.IsAny<SessionId>())).Returns(Observable.Return(session));
    m.Setup(s => s.GetSession(It.IsAny<SessionId>())).Returns(session);
    m.Setup(s => s.ObserveSessionList()).Returns(new BehaviorSubject<IReadOnlyList<SessionId>>(new List<SessionId>()));
    m.Setup(s => s.UpdateSession(It.IsAny<SessionId>(), It.IsAny<Action<T>>()))
      .Callback<SessionId, Action<T>>((_, a) => a(session));
    return m;
  }

  private static Mock<ITranslationService> Translation()
  {
    var t = new Mock<ITranslationService>();
    t.Setup(x => x.CultureChanged).Returns(Observable.Empty<System.Globalization.CultureInfo>());
    return t;
  }

  [Fact]
  public void TimelineTab_RereadsTheProposedRange_OnRestored()
  {
    var schedulers = new TestSchedulerProvider();
    var runtime = new Mock<INativeRuntimeService>();
    var config = new CometConfigService(runtime.Object, schedulers);
    var timeline = new TimelineService(runtime.Object, schedulers, config, new BreadcrumbService());
    var session = new TimelineSession();
    var vm = new TimelineTabViewModel(Translation().Object, schedulers, SessionService(session).Object,
      SessionService(new CometSession()).Object, timeline, runtime.Object, new Mock<ICometMessenger>().Object);
    Assert.Equal("2025-10-01T00:00:00Z", vm.StartEpoch);

    session.ProposedStartEpoch = "2026-01-01T00:00:00Z";
    session.ProposedEndEpoch = "2026-02-01T00:00:00Z";
    session.NotifyRestored();
    Assert.Equal("2026-01-01T00:00:00Z", vm.StartEpoch);
    Assert.Equal("2026-02-01T00:00:00Z", vm.EndEpoch);

    vm.Dispose();
    session.ProposedStartEpoch = "2027-01-01T00:00:00Z";
    session.NotifyRestored();
    Assert.Equal("2026-01-01T00:00:00Z", vm.StartEpoch);
  }

  /// A reopened save sits at its saved epoch, paused: the restored timeline shows the engine's
  /// clock (progress, the viewport's date) and the saved speed, so Play is enabled.
  [Fact]
  public void TimelineTab_ShowsTheClockAndTheSavedSpeed_OnRestored()
  {
    var schedulers = new TestSchedulerProvider();
    var runtime = new Mock<INativeRuntimeService>();
    var config = new CometConfigService(runtime.Object, schedulers);
    var timeline = new TimelineService(runtime.Object, schedulers, config, new BreadcrumbService());
    var session = new TimelineSession();
    var vm = new TimelineTabViewModel(Translation().Object, schedulers, SessionService(session).Object,
      SessionService(new CometSession()).Object, timeline, runtime.Object, new Mock<ICometMessenger>().Object);
    Assert.Null(vm.SelectedSpeed);
    Assert.Equal(0.0, vm.Progress);

    // the save: committed 2024-07-15 .. 2024-08-15, the clock at the end, 1 day/s
    session.CommittedStartEpoch = "2024-07-15T00:00:00Z";
    session.CommittedEndEpoch = "2024-08-15T00:00:00Z";
    session.Speed = (int)TimelineTabViewModel.SimulationSpeed.OneDayPerSec;
    TimeUtils.TryParseIso8601(session.CommittedEndEpoch, out var end);
    var (cent, ns) = TimeUtils.ToTdbParts(end);
    runtime.Setup(r => r.GetSimulationClock(out cent, out ns)).Returns(true);
    session.NotifyRestored();

    Assert.Equal(TimelineTabViewModel.SimulationSpeed.OneDayPerSec, vm.SelectedSpeed);
    Assert.True(vm.PlayPauseCommand.CanExecute(null));
    Assert.Equal(100.0, vm.Progress, 6);
    Assert.StartsWith("2024-08-1", session.CurrentEpochString);
    // a speed the user picks is kept in the session (the next save records it)
    vm.SelectedSpeed = TimelineTabViewModel.SimulationSpeed.OneHourPerSec;
    Assert.Equal((int)TimelineTabViewModel.SimulationSpeed.OneHourPerSec, session.Speed);
  }

  [Fact]
  public void CometTab_RereadsTheRotation_OnRestored()
  {
    var schedulers = new TestSchedulerProvider();
    var runtime = new Mock<INativeRuntimeService>();
    var config = new CometConfigService(runtime.Object, schedulers);
    var breadcrumbs = new BreadcrumbService();
    var timeline = new TimelineService(runtime.Object, schedulers, config, breadcrumbs);
    var storage = new Mock<ILocalStorageService>();
    var session = new CometSession();
    var vm = new CometTabViewModel(Translation().Object, schedulers, SessionService(session).Object,
      new HorizonJplService(null, breadcrumbs, storage.Object), config, timeline, breadcrumbs, storage.Object,
      runtime.Object, SessionService(new ModelSession()).Object, new Mock<ICometMessenger>().Object,
      new Mock<ITabFactory>().Object);

    session.RotPoleRaDeg = 69.3;
    session.RotPoleDecDeg = 64.1;
    session.RotRateDegDay = 649.1;
    session.ReferenceOrbitIndex = 1;
    runtime.Invocations.Clear();
    session.NotifyRestored();
    Assert.Equal(69.3, vm.PoleRaDeg);
    Assert.Equal(64.1, vm.PoleDecDeg);
    Assert.Equal(649.1, vm.RotRateDegDay);
    Assert.Equal(1, vm.ReferenceOrbitIndex);
    // restored values are what native already has: nothing is pushed back
    schedulers.MainThread.AdvanceBy(TimeSpan.FromSeconds(2).Ticks);
    runtime.Verify(r => r.SetBodyRotationalModel(It.IsAny<ulong>(), It.IsAny<BodyRotationalModelDto>()), Times.Never);

    vm.Dispose();
    session.RotPoleRaDeg = 1;
    session.NotifyRestored();
    Assert.Equal(69.3, vm.PoleRaDeg);
  }
}
