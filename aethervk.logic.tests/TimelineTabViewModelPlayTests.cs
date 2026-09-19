using System;
using System.Reactive.Subjects;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using AetherVk.Logic.Tests.Mocks;
using Moq;
using Xunit;
using Microsoft.Reactive.Testing;
using static AetherVk.Logic.ViewModels.TimelineTabViewModel;

namespace AetherVk.Logic.Tests;

public class TimelineTabViewModelPlayTests
{
    private readonly Mock<ITranslationService> _translationServiceMock;
    private readonly TestSchedulerProvider _schedulerProvider;
    private readonly Mock<ITabStateService<TimelineSession>> _timelineSessionServiceMock;
    private readonly Mock<ITabStateService<CometSession>> _cometSessionServiceMock;
    private readonly Mock<INativeRuntimeService> _runtimeServiceMock;
    private readonly Mock<ICometMessenger> _cometMessengerMock;
    private readonly CometConfigService _cometConfig;
    private readonly BreadcrumbService _breadcrumbService;
    private readonly TimelineService _timelineService;
    private readonly Mock<IUiThreadDispatcher> _dispatcherMock;
    private readonly TimelineSession _session;
    private readonly SessionId _sessionId;

    public TimelineTabViewModelPlayTests()
    {
        _translationServiceMock = new Mock<ITranslationService>();
        _translationServiceMock.Setup(t => t.CultureChanged).Returns(System.Reactive.Linq.Observable.Empty<System.Globalization.CultureInfo>());
        _schedulerProvider = new TestSchedulerProvider();
        _timelineSessionServiceMock = new Mock<ITabStateService<TimelineSession>>();
        _timelineSessionServiceMock.Setup(s => s.ActiveSessionIds).Returns(new System.Collections.ObjectModel.ObservableCollection<SessionId> { new SessionId(typeof(TimelineSession), 1) });
        _timelineSessionServiceMock.Setup(s => s.ObserveSessionList()).Returns(new System.Reactive.Subjects.BehaviorSubject<System.Collections.Generic.IReadOnlyList<SessionId>>(new System.Collections.Generic.List<SessionId>()));
        _cometSessionServiceMock = new Mock<ITabStateService<CometSession>>();
        _cometSessionServiceMock.Setup(s => s.ActiveSessionIds).Returns(new System.Collections.ObjectModel.ObservableCollection<SessionId> { new SessionId(typeof(CometSession), 1) });
        _cometSessionServiceMock.Setup(x => x.ObserveSession(It.IsAny<SessionId>())).Returns(System.Reactive.Linq.Observable.Return(new CometSession()));
        _cometSessionServiceMock.Setup(s => s.ObserveSessionList()).Returns(new System.Reactive.Subjects.BehaviorSubject<System.Collections.Generic.IReadOnlyList<SessionId>>(new System.Collections.Generic.List<SessionId>()));
        _runtimeServiceMock = new Mock<INativeRuntimeService>();
        _cometMessengerMock = new Mock<ICometMessenger>();
        
        _cometConfig = new CometConfigService(_runtimeServiceMock.Object, _schedulerProvider);
            
        _dispatcherMock = new Mock<IUiThreadDispatcher>();
        _breadcrumbService = new BreadcrumbService();

        _timelineService = new TimelineService(
            _runtimeServiceMock.Object,
            _schedulerProvider,
            _cometConfig,
            _breadcrumbService);

        _session = new TimelineSession
        {
            CommittedStartEpoch = "2020-01-01T00:00:00Z",
            CommittedEndEpoch = "2020-02-01T00:00:00Z"
        };
        
        _sessionId = new SessionId(typeof(TimelineSession), 1);
        
        // Ensure HasCommittedState is true so commands work
        _timelineSessionServiceMock.Setup(x => x.GetSession(It.IsAny<SessionId>())).Returns(_session);
        _timelineSessionServiceMock.Setup(x => x.ObserveSession(It.IsAny<SessionId>())).Returns(System.Reactive.Linq.Observable.Return(_session));
        _timelineSessionServiceMock.Setup(x => x.UpdateSession(It.IsAny<SessionId>(), It.IsAny<Action<TimelineSession>>()))
            .Callback<SessionId, Action<TimelineSession>>((id, action) => action(_session));
    }

    [Fact]
    public void PlayPauseCommand_RequiresSelectedSpeed()
    {
        var vm = new TimelineTabViewModel(
            _translationServiceMock.Object,
            _schedulerProvider,
            _timelineSessionServiceMock.Object,
            _cometSessionServiceMock.Object,
            _timelineService,
            _runtimeServiceMock.Object,
            _cometMessengerMock.Object);
            
        vm.CurrentSession = _session;

        Assert.False(vm.PlayPauseCommand.CanExecute(null));

        vm.SelectedSpeed = TimelineTabViewModel.SimulationSpeed.OneHourPerSec;
        Assert.True(vm.PlayPauseCommand.CanExecute(null));
    }

    [Fact]
    public void ResetCommand_SetsIsPlayingFalse_AndProgressZero()
    {
        var vm = new TimelineTabViewModel(
            _translationServiceMock.Object,
            _schedulerProvider,
            _timelineSessionServiceMock.Object,
            _cometSessionServiceMock.Object,
            _timelineService,
            _runtimeServiceMock.Object,
            _cometMessengerMock.Object);
            
        vm.CurrentSession = _session;
        vm.IsPlaying = true;
        vm.Progress = 50.0;
        
        _runtimeServiceMock.Setup(r => r.ResetSimulationSync()).Returns(true);

        vm.ResetCommand.Execute(null);

        Assert.False(vm.IsPlaying);
        Assert.Equal(0.0, vm.Progress);
        Assert.Empty(_session.CurrentEpochString);
        _runtimeServiceMock.Verify(r => r.ResetSimulationSync(), Times.Once);
    }

    [Fact]
    public void IsPlayingTrue_SnapshotsSession_AndPlays()
    {
        var vm = new TimelineTabViewModel(
            _translationServiceMock.Object,
            _schedulerProvider,
            _timelineSessionServiceMock.Object,
            _cometSessionServiceMock.Object,
            _timelineService,
            _runtimeServiceMock.Object,
            _cometMessengerMock.Object);
            
        vm.CurrentSession = _session;
        vm.SelectedSpeed = TimelineTabViewModel.SimulationSpeed.OneDayPerSec;
        
        _runtimeServiceMock.Setup(r => r.SnapshotSceneSync()).Returns(true);
        _runtimeServiceMock.Setup(r => r.StartSimulation((int)TimelineTabViewModel.SimulationSpeed.OneDayPerSec)).Returns(true);

        vm.IsPlaying = true;

        Assert.Equal("2020-01-01T00:00:00Z", _session.SnapshotStartEpoch);
        Assert.Equal("2020-02-01T00:00:00Z", _session.SnapshotEndEpoch);
        _runtimeServiceMock.Verify(r => r.SnapshotSceneSync(), Times.Once);
        _runtimeServiceMock.Verify(r => r.StartSimulation((int)TimelineTabViewModel.SimulationSpeed.OneDayPerSec), Times.Once);
    }
}
