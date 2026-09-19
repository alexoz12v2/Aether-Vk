using System;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using AetherVk.Logic.Tests.Mocks;
using Moq;
using Xunit;
using Microsoft.Reactive.Testing;

namespace AetherVk.Logic.Tests;

public class SimulationLockTests
{
    private readonly Mock<ITranslationService> _translationServiceMock;
    private readonly TestSchedulerProvider _schedulerProvider;
    private readonly Mock<ITabStateService<ModelSession>> _modelSessionServiceMock;
    private readonly Mock<ITabStateService<CometSession>> _cometSessionServiceMock;
    private readonly Mock<INativeRuntimeService> _runtimeServiceMock;
    private readonly Mock<ICometMessenger> _cometMessengerMock;
    private readonly Mock<IPlatformWindowService> _platformWindowServiceMock;
    private readonly CometConfigService _cometConfig;
    private readonly BreadcrumbService _breadcrumbService;
    private readonly TimelineService _timelineService;
    private readonly Mock<IUiThreadDispatcher> _dispatcherMock;
    private readonly HorizonJplService _jpl;
    private readonly Mock<ILocalStorageService> _storageMock;
    private readonly Mock<ITabFactory> _tabFactoryMock;

    public SimulationLockTests()
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
            
        _breadcrumbService = new BreadcrumbService();
        _storageMock = new Mock<ILocalStorageService>();
        _jpl = new HorizonJplService(null, _breadcrumbService, _storageMock.Object);
        _tabFactoryMock = new Mock<ITabFactory>();

        _timelineService = new TimelineService(
            _runtimeServiceMock.Object,
            _schedulerProvider,
            _cometConfig,
            _breadcrumbService);
    }

    [Fact]
    public void CometTab_SyncsIsSimulationRunning_FromTimelineService()
    {
        var vm = new CometTabViewModel(
            _translationServiceMock.Object,
            _schedulerProvider,
            _cometSessionServiceMock.Object,
            _jpl,
            _cometConfig,
            _timelineService,
            _breadcrumbService,
            _storageMock.Object,
            _runtimeServiceMock.Object,
            _modelSessionServiceMock.Object,
            _cometMessengerMock.Object,
            _tabFactoryMock.Object);

        Assert.False(vm.IsSimulationRunning);

        _runtimeServiceMock.Setup(r => r.StartSimulation(It.IsAny<int>())).Returns(true);
        _runtimeServiceMock.Setup(r => r.SnapshotSceneSync()).Returns(true);
        _timelineService.Play(1);
        
        _schedulerProvider.MainThread.Start();
        Assert.True(vm.IsSimulationRunning);

        _runtimeServiceMock.Setup(r => r.PauseSimulationSync()).Returns(true);
        _timelineService.Pause();
        
        _schedulerProvider.MainThread.Start();
        Assert.False(vm.IsSimulationRunning);
    }
    
    [Fact]
    public void ModelTab_SyncsIsSimulationRunning_FromTimelineService()
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

        Assert.False(vm.IsSimulationRunning);

        _runtimeServiceMock.Setup(r => r.StartSimulation(It.IsAny<int>())).Returns(true);
        _runtimeServiceMock.Setup(r => r.SnapshotSceneSync()).Returns(true);
        _timelineService.Play(1);
        
        _schedulerProvider.MainThread.Start();
        Assert.True(vm.IsSimulationRunning);

        _runtimeServiceMock.Setup(r => r.PauseSimulationSync()).Returns(true);
        _timelineService.Pause();
        
        _schedulerProvider.MainThread.Start();
        Assert.False(vm.IsSimulationRunning);
    }
}
