using System;
using System.Threading;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using AetherVk.Logic.Tests.Mocks;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// Run to End (">>"): pauses, seeks to the committed end through SeekEpochSync (no Play, no
/// simulated frames), flags the overlay spinner while the seek blocks, and leaves the timeline
/// paused at 100 % with the end epoch published.
public class TimelineTabViewModelRunToEndTests
{
    private delegate bool ClockFn(out short centuries, out ulong nanoseconds);

    [Fact]
    public async Task RunToEnd_SeeksToCommittedEnd_ShowsSpinner_AndEndsPaused()
    {
        var translation = new Mock<ITranslationService>();
        translation.Setup(t => t.CultureChanged).Returns(System.Reactive.Linq.Observable.Empty<System.Globalization.CultureInfo>());
        var schedulers = new TestSchedulerProvider();
        var sessions = new Mock<ITabStateService<TimelineSession>>();
        sessions.Setup(s => s.ActiveSessionIds).Returns(new System.Collections.ObjectModel.ObservableCollection<SessionId> { new SessionId(typeof(TimelineSession), 1) });
        sessions.Setup(s => s.ObserveSessionList()).Returns(new System.Reactive.Subjects.BehaviorSubject<System.Collections.Generic.IReadOnlyList<SessionId>>(new System.Collections.Generic.List<SessionId>()));
        var comets = new Mock<ITabStateService<CometSession>>();
        comets.Setup(s => s.ActiveSessionIds).Returns(new System.Collections.ObjectModel.ObservableCollection<SessionId> { new SessionId(typeof(CometSession), 1) });
        comets.Setup(x => x.ObserveSession(It.IsAny<SessionId>())).Returns(System.Reactive.Linq.Observable.Return(new CometSession()));
        comets.Setup(s => s.ObserveSessionList()).Returns(new System.Reactive.Subjects.BehaviorSubject<System.Collections.Generic.IReadOnlyList<SessionId>>(new System.Collections.Generic.List<SessionId>()));
        var runtime = new Mock<INativeRuntimeService>();
        var timeline = new TimelineService(runtime.Object, schedulers, new CometConfigService(runtime.Object, schedulers), new BreadcrumbService());

        var session = new TimelineSession
        {
            CommittedStartEpoch = "2020-01-01T00:00:00Z",
            CommittedEndEpoch = "2020-02-01T00:00:00Z"
        };
        sessions.Setup(x => x.GetSession(It.IsAny<SessionId>())).Returns(session);
        sessions.Setup(x => x.ObserveSession(It.IsAny<SessionId>())).Returns(System.Reactive.Linq.Observable.Return(session));
        sessions.Setup(x => x.UpdateSession(It.IsAny<SessionId>(), It.IsAny<Action<TimelineSession>>()))
            .Callback<SessionId, Action<TimelineSession>>((id, action) => action(session));

        Assert.True(TimeUtils.TryParseIso8601(session.CommittedEndEpoch, out var endDt));
        var end = TimeUtils.ToTdbParts(endDt);
        var past = TimeUtils.ToTdbParts(endDt.AddDays(1));

        // the seek blocks until released: the spinner flag must be up meanwhile
        using var release = new ManualResetEventSlim(false);
        bool spinnerDuringSeek = false;
        runtime.Setup(r => r.PauseSimulationSync()).Returns(true);
        runtime.Setup(r => r.SeekEpochSync(It.IsAny<short>(), It.IsAny<ulong>()))
            .Returns<short, ulong>((c, ns) =>
            {
                spinnerDuringSeek = session.IsRunningToEnd;
                release.Wait(TimeSpan.FromSeconds(10));
                return true;
            });
        runtime.Setup(r => r.GetSimulationClock(out It.Ref<short>.IsAny, out It.Ref<ulong>.IsAny))
            .Returns(new ClockFn((out short c, out ulong ns) => { c = end.centuries; ns = end.nanoseconds; return true; }));

        var vm = new TimelineTabViewModel(translation.Object, schedulers, sessions.Object, comets.Object,
            timeline, runtime.Object, new Mock<ICometMessenger>().Object);
        vm.CurrentSession = session;

        var run = vm.RunToEndCommand.ExecuteAsync(null);
        Assert.False(vm.IsPlaying);
        release.Set();
        await run;

        // exactly the end (it used to aim a day past it and trust the engine's clamp, which lands
        // on whatever end the engine holds: a stale range sent the clock past the user's end)
        runtime.Verify(r => r.SeekEpochSync(end.centuries, end.nanoseconds), Times.Once);
        runtime.Verify(r => r.SeekEpochSync(past.centuries, past.nanoseconds), Times.Never);
        runtime.Verify(r => r.StartSimulation(It.IsAny<int>()), Times.Never);
        Assert.True(spinnerDuringSeek);
        Assert.False(session.IsRunningToEnd);
        Assert.False(vm.IsPlaying);
        Assert.Equal(100.0, vm.Progress);
        Assert.Equal(endDt.ToString("yyyy-MM-dd HH:mm") + " (TDB)", session.CurrentEpochString);
    }

    [Fact]
    public void RunToEndTarget_PrefersTheEngineConfirmedEnd_OverTheCommittedText()
    {
        Assert.True(TimeUtils.TryParseIso8601("2021-12-28T00:00:00Z", out var userEnd));
        var end = TimeUtils.ToTdbParts(userEnd);
        var start = TimeUtils.ToTdbParts(userEnd.AddDays(-13));
        var confirmed = new TimeRange(start.centuries, start.nanoseconds, end.centuries, end.nanoseconds);

        // the text may say something else (an unsaved edit): the engine's end wins
        Assert.Equal(end, RunToEndTarget(confirmed, "2022-01-27T00:00:00Z"));
        // no confirmation yet: the committed text, exactly (no day added)
        Assert.Equal(end, RunToEndTarget(null, "2021-12-28T00:00:00Z"));
        Assert.Null(RunToEndTarget(null, ""));
    }

    private static (short, ulong)? RunToEndTarget(TimeRange? r, string text) =>
        TimelineTabViewModel.RunToEndTarget(r, text) is var (c, n) ? (c, n) : null;
}
