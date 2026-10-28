using System;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using AetherVk.Logic.Tests.Mocks;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// The timeline's epoch box: shows the engine clock (as the viewport does, to the second) and
/// seeks to a typed epoch inside the committed range.
public class TimelineTabViewModelEpochBoxTests
{
    private delegate bool ClockFn(out short centuries, out ulong nanoseconds);

    private sealed class Rig
    {
        public readonly Mock<INativeRuntimeService> Runtime = new();
        public readonly TimelineSession Session;
        public readonly TimelineService Timeline;
        public readonly TimelineTabViewModel Vm;
        public readonly TestSchedulerProvider Schedulers = new();
        public (short C, ulong N) Clock;

        public Rig(bool committed = true)
        {
            var translation = new Mock<ITranslationService>();
            translation.Setup(t => t.CultureChanged).Returns(System.Reactive.Linq.Observable.Empty<System.Globalization.CultureInfo>());
            var schedulers = Schedulers;
            var sessions = new Mock<ITabStateService<TimelineSession>>();
            sessions.Setup(s => s.ActiveSessionIds).Returns(new System.Collections.ObjectModel.ObservableCollection<SessionId> { new SessionId(typeof(TimelineSession), 1) });
            sessions.Setup(s => s.ObserveSessionList()).Returns(new System.Reactive.Subjects.BehaviorSubject<System.Collections.Generic.IReadOnlyList<SessionId>>(new System.Collections.Generic.List<SessionId>()));
            var comets = new Mock<ITabStateService<CometSession>>();
            comets.Setup(s => s.ActiveSessionIds).Returns(new System.Collections.ObjectModel.ObservableCollection<SessionId> { new SessionId(typeof(CometSession), 1) });
            comets.Setup(x => x.ObserveSession(It.IsAny<SessionId>())).Returns(System.Reactive.Linq.Observable.Return(new CometSession()));
            comets.Setup(s => s.ObserveSessionList()).Returns(new System.Reactive.Subjects.BehaviorSubject<System.Collections.Generic.IReadOnlyList<SessionId>>(new System.Collections.Generic.List<SessionId>()));
            Timeline = new TimelineService(Runtime.Object, schedulers, new CometConfigService(Runtime.Object, schedulers), new BreadcrumbService());

            Session = committed
                ? new TimelineSession { CommittedStartEpoch = "2020-01-01T00:00:00Z", CommittedEndEpoch = "2020-02-01T00:00:00Z" }
                : new TimelineSession();
            sessions.Setup(x => x.GetSession(It.IsAny<SessionId>())).Returns(Session);
            sessions.Setup(x => x.ObserveSession(It.IsAny<SessionId>())).Returns(System.Reactive.Linq.Observable.Return(Session));
            sessions.Setup(x => x.UpdateSession(It.IsAny<SessionId>(), It.IsAny<Action<TimelineSession>>()))
                .Callback<SessionId, Action<TimelineSession>>((id, action) => action(Session));

            Clock = Parts(new DateTimeOffset(2020, 1, 1, 0, 0, 0, TimeSpan.Zero));
            Runtime.Setup(r => r.PauseSimulationSync()).Returns(true);
            Runtime.Setup(r => r.GetSimulationClock(out It.Ref<short>.IsAny, out It.Ref<ulong>.IsAny))
                .Returns(new ClockFn((out short c, out ulong ns) => { c = Clock.C; ns = Clock.N; return true; }));
            // the engine lands where it was asked
            Runtime.Setup(r => r.SeekEpochSync(It.IsAny<short>(), It.IsAny<ulong>()))
                .Returns<short, ulong>((c, ns) => { Clock = (c, ns); return true; });

            Vm = new TimelineTabViewModel(translation.Object, schedulers, sessions.Object, comets.Object,
                Timeline, Runtime.Object, new Mock<ICometMessenger>().Object);
            Vm.CurrentSession = Session;
        }
    }

    private static (short, ulong) Parts(DateTimeOffset t) => TimeUtils.ToTaiParts(t);

    [Fact]
    public void Box_ShowsTheEngineClock_EvenAtTheStart()
    {
        var rig = new Rig();
        // the viewport label is empty at the start; the box always shows the epoch
        Assert.Equal("2020-01-01 00:00:00", rig.Vm.CurrentEpochText);
        Assert.False(rig.Vm.HasCurrentEpochError);
    }

    [Fact]
    public async Task TypedEpoch_InRange_SeeksExactlyThere_AndRefreshesTheBox()
    {
        var rig = new Rig();
        rig.Vm.IsEditingCurrentEpoch = true; // the user is typing, then presses Enter
        rig.Vm.CurrentEpochText = "2020-01-15 06:30:15";
        await rig.Vm.SeekToEpochCommand.ExecuteAsync(null);

        var target = Parts(new DateTimeOffset(2020, 1, 15, 6, 30, 15, TimeSpan.Zero));
        rig.Runtime.Verify(r => r.SeekEpochSync(target.Item1, target.Item2), Times.Once);
        rig.Runtime.Verify(r => r.StartSimulation(It.IsAny<int>()), Times.Never);
        Assert.False(rig.Vm.IsPlaying);
        Assert.False(rig.Session.IsRunningToEnd);
        Assert.False(rig.Vm.HasCurrentEpochError);
        Assert.Equal("2020-01-15 06:30:15", rig.Vm.CurrentEpochText);
        Assert.Equal("2020-01-15 06:30", rig.Session.CurrentEpochString);
        Assert.InRange(rig.Vm.Progress, 46.0, 46.1); // 14.27 d of 31
    }

    [Theory]
    [InlineData("not a date")]
    [InlineData("")]
    [InlineData("2020-13-45 00:00:00")]
    public async Task TypedGarbage_ShowsAnError_AndDoesNotSeek(string text)
    {
        var rig = new Rig();
        rig.Vm.CurrentEpochText = text;
        await rig.Vm.SeekToEpochCommand.ExecuteAsync(null);
        rig.Runtime.Verify(r => r.SeekEpochSync(It.IsAny<short>(), It.IsAny<ulong>()), Times.Never);
        Assert.True(rig.Vm.HasCurrentEpochError);
        Assert.Contains("yyyy-MM-dd HH:mm:ss", rig.Vm.CurrentEpochError);
    }

    [Theory]
    [InlineData("2019-12-31 23:59:59")]
    [InlineData("2020-02-01 00:00:01")]
    public async Task TypedEpoch_OutsideTheCommittedRange_IsRefused(string text)
    {
        var rig = new Rig();
        rig.Vm.CurrentEpochText = text;
        await rig.Vm.SeekToEpochCommand.ExecuteAsync(null);
        rig.Runtime.Verify(r => r.SeekEpochSync(It.IsAny<short>(), It.IsAny<ulong>()), Times.Never);
        Assert.Equal("Outside the timeline (2020-01-01 00:00:00 to 2020-02-01 00:00:00).", rig.Vm.CurrentEpochError);
    }

    [Fact]
    public async Task TheRangeEnds_AreAccepted_AndAnErrorClearsOnTheNextGoodSeek()
    {
        var rig = new Rig();
        rig.Vm.CurrentEpochText = "nope";
        await rig.Vm.SeekToEpochCommand.ExecuteAsync(null);
        Assert.True(rig.Vm.HasCurrentEpochError);

        rig.Vm.CurrentEpochText = "2020-02-01 00:00:00";
        await rig.Vm.SeekToEpochCommand.ExecuteAsync(null);
        var end = Parts(new DateTimeOffset(2020, 2, 1, 0, 0, 0, TimeSpan.Zero));
        rig.Runtime.Verify(r => r.SeekEpochSync(end.Item1, end.Item2), Times.Once);
        Assert.False(rig.Vm.HasCurrentEpochError);
        Assert.Equal(100.0, rig.Vm.Progress);
    }

    [Fact]
    public async Task WithoutACommittedRange_NothingIsSought()
    {
        var rig = new Rig(committed: false);
        Assert.False(rig.Vm.HasCommittedState);
        rig.Vm.CurrentEpochText = "2020-01-15 00:00:00";
        await rig.Vm.SeekToEpochCommand.ExecuteAsync(null);
        rig.Runtime.Verify(r => r.SeekEpochSync(It.IsAny<short>(), It.IsAny<ulong>()), Times.Never);
    }

    /// <summary>A seek from elsewhere (a FITS observation epoch) goes through the shared helper and
    /// refreshes the box; typing is not overwritten by it.</summary>
    [Fact]
    public async Task SharedSeek_RefreshesTheBox_UnlessTheUserIsTyping()
    {
        var rig = new Rig();
        var t1 = Parts(new DateTimeOffset(2020, 1, 20, 12, 0, 0, TimeSpan.Zero));
        await rig.Timeline.SeekAsync(t1.Item1, t1.Item2);
        rig.Schedulers.MainThread.AdvanceBy(1); // Seeked is delivered on the UI thread
        Assert.Equal("2020-01-20 12:00:00", rig.Vm.CurrentEpochText);

        rig.Vm.IsEditingCurrentEpoch = true;
        rig.Vm.CurrentEpochText = "2020-01-2";
        var t2 = Parts(new DateTimeOffset(2020, 1, 25, 0, 0, 0, TimeSpan.Zero));
        await rig.Timeline.SeekAsync(t2.Item1, t2.Item2);
        rig.Schedulers.MainThread.AdvanceBy(1);
        Assert.Equal("2020-01-2", rig.Vm.CurrentEpochText);
        Assert.Equal("2020-01-25 00:00", rig.Session.CurrentEpochString);
    }

    [Fact]
    public void Parse_IsInvariantAndUtc_AndRoundTripsTheDisplay()
    {
        var expected = Parts(new DateTimeOffset(2021, 12, 27, 0, 14, 47, TimeSpan.Zero));
        Assert.True(TimelineTabViewModel.TryParseEpochBox("2021-12-27 00:14:47", out var a));
        Assert.Equal(expected, (a.Centuries, a.Nanoseconds));
        Assert.True(TimelineTabViewModel.TryParseEpochBox(" 2021-12-27T00:14:47Z ", out var b));
        Assert.Equal(expected, (b.Centuries, b.Nanoseconds));
        // an explicit offset is honoured
        Assert.True(TimelineTabViewModel.TryParseEpochBox("2021-12-27T01:14:47+01:00", out var c));
        Assert.Equal(expected, (c.Centuries, c.Nanoseconds));
        Assert.Equal("2021-12-27 00:14:47", TimelineTabViewModel.FormatEpoch(expected.Item1, expected.Item2));
        Assert.False(TimelineTabViewModel.TryParseEpochBox("   ", out _));
    }
}
