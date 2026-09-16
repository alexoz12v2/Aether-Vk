using System;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using AetherVk.Logic.Tests.Mocks;
using Moq;
using Xunit;
using Microsoft.Reactive.Testing;
using System.Reactive.Subjects;
using System.Collections.ObjectModel;
using System.Collections.Generic;
using System.Reactive.Linq;

namespace AetherVk.Logic.Tests;

public class CometTabViewModelDecommitTests
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

    public CometTabViewModelDecommitTests()
    {
        _translationServiceMock = new Mock<ITranslationService>();
        _translationServiceMock.Setup(t => t.CultureChanged).Returns(Observable.Empty<System.Globalization.CultureInfo>());
        _schedulerProvider = new TestSchedulerProvider();
        
        _modelSessionServiceMock = new Mock<ITabStateService<ModelSession>>();
        _modelSessionServiceMock.Setup(s => s.ActiveSessionIds).Returns(new ObservableCollection<SessionId> { new SessionId(typeof(ModelSession), 1) });
        _modelSessionServiceMock.Setup(x => x.ObserveSession(It.IsAny<SessionId>())).Returns(Observable.Return(new ModelSession()));
        _modelSessionServiceMock.Setup(s => s.ObserveSessionList()).Returns(new BehaviorSubject<IReadOnlyList<SessionId>>(new List<SessionId>()));
        
        _cometSessionServiceMock = new Mock<ITabStateService<CometSession>>();
        _cometSessionServiceMock.Setup(s => s.ActiveSessionIds).Returns(new ObservableCollection<SessionId> { new SessionId(typeof(CometSession), 1) });
        _cometSessionServiceMock.Setup(x => x.ObserveSession(It.IsAny<SessionId>())).Returns(Observable.Return(new CometSession()));
        _cometSessionServiceMock.Setup(s => s.ObserveSessionList()).Returns(new BehaviorSubject<IReadOnlyList<SessionId>>(new List<SessionId>()));
        
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
    }

    [Fact]
    public void DecommitCometCommand_ResetsSimulation_IfRunning()
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

        _runtimeServiceMock.Setup(r => r.StartSimulation(It.IsAny<int>())).Returns(true);
        _runtimeServiceMock.Setup(r => r.SnapshotSceneSync()).Returns(true);

        // Start simulation
        _timelineService.Play(1);
        _schedulerProvider.MainThread.Start();
        
        Assert.True(vm.IsSimulationRunning);
        Assert.True(_timelineService.IsSimulationRunningValue);

        // Commit an almanac to enable DecommitComet
        // We set this here because starting the scheduler pumps the initial 'false' from CometConfigService
        vm.IsAlmanacCommitted = true;

        // Decommit
        vm.DecommitCometCommand.Execute(null);

        // Verify ResetSimulationSync was called
        _runtimeServiceMock.Verify(r => r.ResetSimulationSync(), Times.Once);
    }
}
