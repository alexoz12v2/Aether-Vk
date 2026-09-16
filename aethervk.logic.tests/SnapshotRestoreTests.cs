using System;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using AetherVk.Logic.Tests.Mocks;
using Moq;
using Xunit;
using Microsoft.Reactive.Testing;

namespace AetherVk.Logic.Tests;

public class SnapshotRestoreTests
{
    private readonly Mock<ITranslationService> _translationServiceMock;
    private readonly TestSchedulerProvider _schedulerProvider;
    private readonly Mock<ITabStateService<ModelSession>> _modelSessionServiceMock;
    private readonly Mock<ITabStateService<CometSession>> _cometSessionServiceMock;
    private readonly Mock<INativeRuntimeService> _runtimeServiceMock;
    private readonly Mock<ICometMessenger> _cometMessengerMock;
    private readonly Mock<IPlatformWindowService> _platformWindowServiceMock;
    private readonly CometConfigService _cometConfig;
    private readonly Mock<BreadcrumbService> _breadcrumbServiceMock;
    private readonly TimelineService _timelineService;
    private readonly Mock<IUiThreadDispatcher> _dispatcherMock;
    private readonly HorizonJplService _jpl;
    private readonly Mock<ILocalStorageService> _storageMock;
    private readonly Mock<ITabFactory> _tabFactoryMock;

    public SnapshotRestoreTests()
    {
        _translationServiceMock = new Mock<ITranslationService>();
        _translationServiceMock.Setup(t => t.CultureChanged).Returns(System.Reactive.Linq.Observable.Empty<System.Globalization.CultureInfo>());
        _schedulerProvider = new TestSchedulerProvider();
        _modelSessionServiceMock = new Mock<ITabStateService<ModelSession>>();
        _modelSessionServiceMock.Setup(s => s.ActiveSessionIds).Returns(new System.Collections.ObjectModel.ObservableCollection<SessionId> { new SessionId(typeof(ModelSession), 1) });
        _modelSessionServiceMock.Setup(x => x.ObserveSession(It.IsAny<SessionId>())).Returns(System.Reactive.Linq.Observable.Return(new ModelSession()));
        _modelSessionServiceMock.Setup(s => s.ObserveSessionList()).Returns(new System.Reactive.Subjects.BehaviorSubject<System.Collections.Generic.IReadOnlyList<SessionId>>(new System.Collections.Generic.List<SessionId>()));
        _cometSessionServiceMock = new Mock<ITabStateService<CometSession>>();
        _cometSessionServiceMock.Setup(s => s.ActiveSessionIds).Returns(new System.Collections.ObjectModel.ObservableCollection<SessionId> { new SessionId(typeof(CometSession), 1) });
        _cometSessionServiceMock.Setup(x => x.ObserveSession(It.IsAny<SessionId>())).Returns(System.Reactive.Linq.Observable.Return(new CometSession()));
        _cometSessionServiceMock.Setup(s => s.ObserveSessionList()).Returns(new System.Reactive.Subjects.BehaviorSubject<System.Collections.Generic.IReadOnlyList<SessionId>>(new System.Collections.Generic.List<SessionId>()));
        _runtimeServiceMock = new Mock<INativeRuntimeService>();
        _cometMessengerMock = new Mock<ICometMessenger>();
        _platformWindowServiceMock = new Mock<IPlatformWindowService>();
        _dispatcherMock = new Mock<IUiThreadDispatcher>();
        
        _cometConfig = new CometConfigService(_runtimeServiceMock.Object, _schedulerProvider);
            
        _breadcrumbServiceMock = new Mock<BreadcrumbService>(_dispatcherMock.Object);
        _storageMock = new Mock<ILocalStorageService>();
        _jpl = new HorizonJplService(null, _breadcrumbServiceMock.Object, _storageMock.Object);
        _tabFactoryMock = new Mock<ITabFactory>();

        _timelineService = new TimelineService(
            _runtimeServiceMock.Object,
            _schedulerProvider,
            _cometConfig,
            _breadcrumbServiceMock.Object);
            
        // Make dispatcher execute immediately for tests
        _dispatcherMock.Setup(d => d.Dispatch(It.IsAny<Action>())).Callback<Action>(a => a());
    }

    [Fact]
    public void ChangingCometSettings_WhilePaused_TriggersSnapshotRestore()
    {
        var vm = new CometTabViewModel(
            _translationServiceMock.Object,
            _schedulerProvider,
            _cometSessionServiceMock.Object,
            _jpl,
            _cometConfig,
            _timelineService,
            _breadcrumbServiceMock.Object,
            _storageMock.Object,
            _runtimeServiceMock.Object,
            _modelSessionServiceMock.Object,
            _cometMessengerMock.Object,
            _tabFactoryMock.Object);

        // Simulate Play
        _runtimeServiceMock.Setup(r => r.StartSimulation(It.IsAny<int>())).Returns(true);
        _runtimeServiceMock.Setup(r => r.SnapshotSceneSync()).Returns(true);
        _timelineService.Play(1);
        _schedulerProvider.MainThread.Start();
        
        // Simulate Pause
        _runtimeServiceMock.Setup(r => r.PauseSimulationSync()).Returns(true);
        _timelineService.Pause();
        _schedulerProvider.MainThread.Start();
        
        // At this point, _snapshotExists = true, IsSimulationRunning = false.
        
        _runtimeServiceMock.Setup(r => r.RestoreSnapshotSync()).Returns(true);
        
        // Trigger a setting change
        vm.PoleRaDeg = 45;
        
        // Assert
        _runtimeServiceMock.Verify(r => r.RestoreSnapshotSync(), Times.Once);
        
        // Changing it again shouldn't trigger another restore because _snapshotExists is false now
        vm.PoleDecDeg = 20;
        _runtimeServiceMock.Verify(r => r.RestoreSnapshotSync(), Times.Once); // Still 1
    }
    
    [Fact]
    public void ChangingModelSettings_WhilePaused_TriggersSnapshotRestore()
    {
        var vm = new ModelTabViewModel(
            _translationServiceMock.Object,
            _schedulerProvider,
            _modelSessionServiceMock.Object,
            _cometSessionServiceMock.Object,
            _cometConfig,
            _runtimeServiceMock.Object,
            _dispatcherMock.Object,
            _cometMessengerMock.Object,
            _platformWindowServiceMock.Object,
            _timelineService);

        // Simulate Play
        _runtimeServiceMock.Setup(r => r.StartSimulation(It.IsAny<int>())).Returns(true);
        _runtimeServiceMock.Setup(r => r.SnapshotSceneSync()).Returns(true);
        _timelineService.Play(1);
        _schedulerProvider.MainThread.Start();
        
        // Simulate Pause
        _runtimeServiceMock.Setup(r => r.PauseSimulationSync()).Returns(true);
        _timelineService.Pause();
        _schedulerProvider.MainThread.Start();
        
        // At this point, _snapshotExists = true, IsSimulationRunning = false.
        
        _runtimeServiceMock.Setup(r => r.RestoreSnapshotSync()).Returns(true);
        
        // Trigger a setting change
        vm.ManualNucleusRadiusKm = 15f;
        
        // Assert
        _runtimeServiceMock.Verify(r => r.RestoreSnapshotSync(), Times.Once);
    }
}
