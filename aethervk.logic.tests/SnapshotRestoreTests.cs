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
    private readonly BreadcrumbService _breadcrumbService;
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
            
        _breadcrumbService = new BreadcrumbService();
        _storageMock = new Mock<ILocalStorageService>();
        _jpl = new HorizonJplService(null, _breadcrumbService, _storageMock.Object);
        _tabFactoryMock = new Mock<ITabFactory>();

        _timelineService = new TimelineService(
            _runtimeServiceMock.Object,
            _schedulerProvider,
            _cometConfig,
            _breadcrumbService);
            
        // Make dispatcher execute immediately for tests
        _dispatcherMock.Setup(d => d.Dispatch(It.IsAny<Action>())).Callback<Action>(a => a());
    }

    /// A paused edit after Play stays at the current date: the Comet tab's rotation edits no
    /// longer restore the scene snapshot taken at Play (the restore rewound the clock and left
    /// the engine without the comet's state, so the model did not show again until a seek or a
    /// reopen; the engine re-emits the dust of a changed rotation on its own).
    [Fact]
    public void ChangingCometSettings_WhilePaused_DoesNotRestoreSnapshot()
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

        // Simulate Play
        _runtimeServiceMock.Setup(r => r.StartSimulation(It.IsAny<int>())).Returns(true);
        _runtimeServiceMock.Setup(r => r.SnapshotSceneSync()).Returns(true);
        _timelineService.Play(1);
        _schedulerProvider.MainThread.Start();
        _runtimeServiceMock.Verify(r => r.SnapshotSceneSync(), Times.Once); // Play still snapshots

        // Simulate Pause
        _runtimeServiceMock.Setup(r => r.PauseSimulationSync()).Returns(true);
        _timelineService.Pause();
        _schedulerProvider.MainThread.Start();
        Assert.False(vm.IsSimulationRunning);

        _runtimeServiceMock.Setup(r => r.RestoreSnapshotSync()).Returns(true);

        // Paused edits after a Play: no restore, the clock stays where it is
        vm.PoleRaDeg = 45;
        _runtimeServiceMock.Verify(r => r.RestoreSnapshotSync(), Times.Never);
        vm.PoleDecDeg = 20;
        _runtimeServiceMock.Verify(r => r.RestoreSnapshotSync(), Times.Never);
        _runtimeServiceMock.Verify(r => r.ResetSimulationSync(), Times.Never);
    }

    /// The nucleus radius (Comet tab, feedback item 10b) edited after Play while paused: no
    /// restore either.
    [Fact]
    public void ChangingNucleusRadius_InCometTab_WhilePaused_DoesNotRestoreSnapshot()
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

        // Simulate Play
        _runtimeServiceMock.Setup(r => r.StartSimulation(It.IsAny<int>())).Returns(true);
        _runtimeServiceMock.Setup(r => r.SnapshotSceneSync()).Returns(true);
        _timelineService.Play(1);
        _schedulerProvider.MainThread.Start();

        // Simulate Pause
        _runtimeServiceMock.Setup(r => r.PauseSimulationSync()).Returns(true);
        _timelineService.Pause();
        _schedulerProvider.MainThread.Start();

        _runtimeServiceMock.Setup(r => r.RestoreSnapshotSync()).Returns(true);

        vm.ManualNucleusRadiusKm = 15f;
        _runtimeServiceMock.Verify(r => r.RestoreSnapshotSync(), Times.Never);
        vm.ManualNucleusRadiusKm = 16f;
        _runtimeServiceMock.Verify(r => r.RestoreSnapshotSync(), Times.Never);
    }

    /// A jet edited while paused after a Play (visual or physical) reaches the native runtime
    /// through `ModifyParticleSystem` and never restores the snapshot: the engine re-emits the
    /// changed jet's dust at the paused date.
    [Fact]
    public void JetEdits_WhilePaused_ReachTheRuntime_WithoutRestoringTheSnapshot()
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

        // a committed comet with a live jet
        ulong psId = 7;
        _runtimeServiceMock
            .Setup(r => r.AddFirstParticleSystem(It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out psId))
            .Returns(new ParticleSystemComputedProperties(0.1f, 1f));
        var computed = new ParticleSystemComputedProperties(0.1f, 1f);
        _runtimeServiceMock
            .Setup(r => r.ModifyParticleSystem(It.IsAny<ulong>(), It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out computed))
            .Returns(true);
        // deliver the initial "not committed" emission first (it would reset NativePsId later)
        _schedulerProvider.MainThread.Start();
        // the model session's default manual radius (2 km, edited in the Comet tab) places the jet
        vm.IsCometCommitted = true;
        vm.AddJetCommand.Execute(null);
        var jet = Assert.Single(vm.Jets!);
        Assert.Equal(psId, jet.NativePsId);

        // Play, then Pause: a snapshot exists in the engine
        _runtimeServiceMock.Setup(r => r.StartSimulation(It.IsAny<int>())).Returns(true);
        _runtimeServiceMock.Setup(r => r.SnapshotSceneSync()).Returns(true);
        _timelineService.Play(1);
        _schedulerProvider.MainThread.Start();
        _runtimeServiceMock.Setup(r => r.PauseSimulationSync()).Returns(true);
        _timelineService.Pause();
        _schedulerProvider.MainThread.Start();
        _runtimeServiceMock.Setup(r => r.RestoreSnapshotSync()).Returns(true);
        _runtimeServiceMock.Invocations.Clear();

        // a visual edit
        jet.StreamColor = new System.Numerics.Vector4(1f, 0f, 0f, 1f);
        _schedulerProvider.Background.AdvanceBy(TimeSpan.FromMilliseconds(300).Ticks);

        _runtimeServiceMock.Verify(r => r.RestoreSnapshotSync(), Times.Never);
        _runtimeServiceMock.Verify(
            r => r.ModifyParticleSystem(psId, It.IsAny<ParticleSystemModel>(),
                It.Is<ParticleSystemJet>(j => j.StreamColor.X == 1f && j.StreamColor.Y == 0f),
                out It.Ref<ParticleSystemComputedProperties>.IsAny),
            Times.Once);

        // a physical edit: forwarded too, still no restore
        jet.LatitudeDeg = 12f;
        _schedulerProvider.Background.AdvanceBy(TimeSpan.FromMilliseconds(300).Ticks);
        _runtimeServiceMock.Verify(r => r.RestoreSnapshotSync(), Times.Never);
        _runtimeServiceMock.Verify(
            r => r.ModifyParticleSystem(psId, It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(),
                out It.Ref<ParticleSystemComputedProperties>.IsAny),
            Times.Exactly(2));
        Assert.False(_timelineService.IsSimulationRunningValue);
    }

    /// "show reference-position error" reaches the native runtime and is kept in the session,
    /// without restoring the snapshot (it is a drawing option).
    [Fact]
    public void ShowReferencePositionError_CallsRuntime_AndPersistsInSession()
    {
        var session = new ModelSession();
        _modelSessionServiceMock.Setup(x => x.ObserveSession(It.IsAny<SessionId>()))
            .Returns(System.Reactive.Linq.Observable.Return(session));
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

        vm.ShowReferencePositionError = true;
        _runtimeServiceMock.Verify(r => r.SetReferenceErrorVisible(true), Times.Once);
        Assert.True(session.ShowReferencePositionError);
        _runtimeServiceMock.Verify(r => r.RestoreSnapshotSync(), Times.Never);

        vm.ShowReferencePositionError = false;
        _runtimeServiceMock.Verify(r => r.SetReferenceErrorVisible(false), Times.Once);
        Assert.False(session.ShowReferencePositionError);
    }

    /// Tabs are scoped: a reopened Model tab is a new VM over the same session and must show the
    /// checkbox as the session (and the native runtime) has it.
    [Fact]
    public void ShowReferencePositionError_IsRestored_ByANewVmOverTheSameSession()
    {
        var session = new ModelSession { ShowReferencePositionError = true };
        _modelSessionServiceMock.Setup(x => x.ObserveSession(It.IsAny<SessionId>()))
            .Returns(System.Reactive.Linq.Observable.Return(session));
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

        Assert.True(vm.ShowReferencePositionError);
        Assert.True(session.ShowReferencePositionError);
    }
}
