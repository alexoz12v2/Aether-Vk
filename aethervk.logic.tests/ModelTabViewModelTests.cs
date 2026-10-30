using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Microsoft.Reactive.Testing;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

// Moq requires a named delegate for Callback on methods with out parameters.
internal delegate void ModifyParticleSystemCallback(
  ulong psId,
  ParticleSystemModel psModel,
  ParticleSystemJet psJet,
  out ParticleSystemComputedProperties outPsComputedProps);

/// <summary>
/// Unit tests for <see cref="ModelTabViewModel"/> covering jet management and
/// per-jet grain and activity parameters. Native FFI paths are bypassed via a mocked
/// <see cref="INativeRuntimeService"/>.
/// </summary>
public class ModelTabViewModelTests
{
  // ── helpers ───────────────────────────────────────────────────────────────────

  private sealed class TestSetup
  {
    public ModelTabViewModel Vm { get; }
    public Mock<INativeRuntimeService> Runtime { get; }
    public ModelSession Session { get; }

    public TestSetup(float nucleusRadiusKm = 5.0f, bool cometCommitted = true, ICameraServiceRegistry? cameras = null)
    {
      Runtime = new Mock<INativeRuntimeService>();
      // RegisterExternalStateListener must return a non-null disposable.
      Runtime
        .Setup(r => r.RegisterExternalStateListener(It.IsAny<ExternalStateType>(), It.IsAny<Action<nint>>()))
        .Returns(System.Reactive.Disposables.Disposable.Empty);

      var dispatcher = new Mock<IUiThreadDispatcher>();
      dispatcher.Setup(d => d.Dispatch(It.IsAny<Action>())).Callback<Action>(a => a());
      dispatcher.Setup(d => d.CheckAccess()).Returns(true);

      var schedulers = new Mock<ISchedulerProvider>();
      schedulers
        .Setup(s => s.MainThread)
        .Returns(System.Reactive.Concurrency.ImmediateScheduler.Instance);
      schedulers
        .Setup(s => s.Background)
        .Returns(System.Reactive.Concurrency.ImmediateScheduler.Instance);

      // the manual radius is edited in the Comet tab: the Model tab reads it from the session
      Session = new ModelSession { ManualNucleusRadiusKm = nucleusRadiusKm };
      var sessionService = new Mock<ITabStateService<ModelSession>>();
      var sessionList = new[] { new SessionId(typeof(ModelSession), 1) };
      sessionService.Setup(s => s.ActiveSessionIds).Returns(sessionList);
      sessionService
        .Setup(s => s.ObserveSessionList())
        .Returns(
          System.Reactive.Linq.Observable
            .Return<IReadOnlyList<SessionId>>(sessionList));
      sessionService
        .Setup(s => s.ObserveSession(It.IsAny<SessionId>()))
        .Returns(System.Reactive.Linq.Observable.Return(Session));
      sessionService
        .Setup(s => s.GetSession(It.IsAny<SessionId>()))
        .Returns(Session);
      sessionService.Setup(s => s.IsExclusive).Returns(true);

      var cometSessionService = new Mock<ITabStateService<CometSession>>();
      cometSessionService
        .Setup(s => s.ActiveSessionIds)
        .Returns(Array.Empty<SessionId>());
      cometSessionService
        .Setup(s => s.ObserveSessionList())
        .Returns(
          System.Reactive.Linq.Observable
            .Return<IReadOnlyList<SessionId>>(Array.Empty<SessionId>()));

      var translationService = new Mock<ITranslationService>();
      translationService
        .Setup(t => t.CultureChanged)
        .Returns(System.Reactive.Linq.Observable.Never<System.Globalization.CultureInfo>());

      // Construct the real CometConfigService with mocked dependencies.
      var cometConfig = new CometConfigService(Runtime.Object, schedulers.Object);

      var cometMessenger = new Mock<ICometMessenger>();

      var breadcrumbService = new BreadcrumbService();
      var timelineService = new TimelineService(
        Runtime.Object,
        schedulers.Object,
        cometConfig,
        breadcrumbService
      );

      Vm = new ModelTabViewModel(
        translationService.Object,
        schedulers.Object,
        sessionService.Object,
        cometSessionService.Object,
        cometConfig,
        Runtime.Object,
        dispatcher.Object,
        cometMessenger.Object,
        new Mock<IPlatformWindowService>().Object,
        timelineService,
        cameras
      ); // Most tests need a committed comet — simulate commitment via the observable property.
      if (cometCommitted)
        Vm.IsCometCommitted = true;
    }
  }

  // ── tests ─────────────────────────────────────────────────────────────────────

  [Fact]
  public void SphereGizmo_Toggle_ReachesNativeAndSession_AndIsReappliedOnCommit()
  {
    var s = new TestSetup(cometCommitted: false);

    // before any commit: stored, and the commit applies it (a commit rebuilds the gizmo shown)
    s.Vm.IsGizmoVisible = false;
    Assert.False(s.Session.SphereGizmoVisible);
    s.Runtime.Invocations.Clear();
    s.Vm.IsCometCommitted = true;
    s.Runtime.Verify(r => r.SetSphereGizmoVisibility(false), Times.Once);

    s.Vm.IsGizmoVisible = true;
    Assert.True(s.Session.SphereGizmoVisible);
    s.Runtime.Verify(r => r.SetSphereGizmoVisibility(true), Times.Once);
  }

  [Fact]
  public void SphereGizmoParts_Toggles_ReachNativeAndSession_AndAreReappliedOnCommit()
  {
    var s = new TestSetup(cometCommitted: false);
    Assert.Equal(AetherVk.Logic.Models.SphereGizmoPart.Default, s.Vm.GizmoParts);
    Assert.False(s.Vm.ShowGizmoVelocity);

    s.Vm.ShowGizmoGrid = false;
    s.Vm.ShowGizmoVelocity = true;
    uint expected = AetherVk.Logic.Models.SphereGizmoPart.All & ~AetherVk.Logic.Models.SphereGizmoPart.Grid;
    Assert.Equal(expected, s.Session.SphereGizmoParts);
    s.Runtime.Verify(r => r.SetSphereGizmoParts(expected), Times.Once);

    s.Runtime.Invocations.Clear();
    s.Vm.IsCometCommitted = true;
    s.Runtime.Verify(r => r.SetSphereGizmoParts(expected), Times.Once);

    // the save file keeps it
    var dto = AetherVk.Logic.SaveFile.SaveIndexMapping.ToDto(s.Session);
    var restored = new ModelSession();
    AetherVk.Logic.SaveFile.SaveIndexMapping.Apply(dto, restored);
    Assert.Equal(expected, restored.SphereGizmoParts);
  }

  /// Feedback item 16: each gizmo part can be shown alone; every toggle maps to its own bit.
  [Theory]
  [InlineData(nameof(ModelTabViewModel.ShowGizmoGrid), AetherVk.Logic.Models.SphereGizmoPart.Grid)]
  [InlineData(nameof(ModelTabViewModel.ShowGizmoBodyAxes), AetherVk.Logic.Models.SphereGizmoPart.BodyAxes)]
  [InlineData(nameof(ModelTabViewModel.ShowGizmoSpinAxis), AetherVk.Logic.Models.SphereGizmoPart.SpinAxis)]
  [InlineData(nameof(ModelTabViewModel.ShowGizmoSun), AetherVk.Logic.Models.SphereGizmoPart.Sun)]
  [InlineData(nameof(ModelTabViewModel.ShowGizmoVelocity), AetherVk.Logic.Models.SphereGizmoPart.Velocity)]
  public void SphereGizmoPart_EachToggle_ShowsItsPartAlone(string toggle, uint bit)
  {
    Assert.Equal(new uint[] { 1u, 2u, 4u, 8u, 16u },
      new[] { AetherVk.Logic.Models.SphereGizmoPart.Grid, AetherVk.Logic.Models.SphereGizmoPart.BodyAxes,
        AetherVk.Logic.Models.SphereGizmoPart.SpinAxis, AetherVk.Logic.Models.SphereGizmoPart.Sun,
        AetherVk.Logic.Models.SphereGizmoPart.Velocity });
    var s = new TestSetup();
    var prop = typeof(ModelTabViewModel).GetProperty(toggle)!;
    s.Vm.GizmoParts = 0;
    s.Runtime.Invocations.Clear();

    prop.SetValue(s.Vm, true);
    Assert.Equal(bit, s.Vm.GizmoParts);
    Assert.Equal(bit, s.Session.SphereGizmoParts);
    s.Runtime.Verify(r => r.SetSphereGizmoParts(bit), Times.Once);
    Assert.True((bool)prop.GetValue(s.Vm)!);

    prop.SetValue(s.Vm, false);
    Assert.Equal(0u, s.Vm.GizmoParts);
  }

  /// <summary>Jets with native ids 11 and 22, the first one selected.</summary>
  private static (JetViewModel A, JetViewModel B) TwoNativeJets(TestSetup s)
  {
    var a = new JetViewModel(10, 20, 30, 2, 0.5f, new System.Numerics.Vector4(1, 1, 1, 1), true, 1u) { NativePsId = 11 };
    var b = new JetViewModel(-5, 120, 30, 2, 0.5f, new System.Numerics.Vector4(1, 1, 1, 1), true, 2u) { NativePsId = 22 };
    s.Session.Jets.Clear();
    s.Session.Jets.Add(a);
    s.Session.Jets.Add(b);
    return (a, b);
  }

  /// <summary>Jet angles: shown on the selected jet only, following the selection; saved.</summary>
  [Fact]
  public void JetAngles_FollowTheSelectedJetOnly_AndAreSaved()
  {
    var s = new TestSetup();
    var (a, b) = TwoNativeJets(s);
    s.Vm.SelectedJet = a;
    s.Runtime.Invocations.Clear();

    s.Vm.ShowJetAngles = true;
    Assert.True(s.Session.ShowJetAngles);
    s.Runtime.Verify(r => r.SetJetAngleVisibility(11ul, true), Times.Once);
    s.Runtime.Verify(r => r.SetJetAngleVisibility(22ul, false), Times.Once);

    s.Runtime.Invocations.Clear();
    s.Vm.SelectedJet = b;
    s.Runtime.Verify(r => r.SetJetAngleVisibility(11ul, false), Times.Once);
    s.Runtime.Verify(r => r.SetJetAngleVisibility(22ul, true), Times.Once);

    s.Runtime.Invocations.Clear();
    s.Vm.ShowJetAngles = false;
    Assert.False(s.Session.ShowJetAngles);
    s.Runtime.Verify(r => r.SetJetAngleVisibility(It.IsAny<ulong>(), true), Times.Never);
    s.Runtime.Verify(r => r.SetJetAngleVisibility(22ul, false), Times.Once);

    // the savefile round trip and a restore into the tab
    s.Session.ShowJetAngles = true;
    var dto = AetherVk.Logic.SaveFile.SaveIndexMapping.ToDto(s.Session);
    Assert.True(dto.ShowJetAngles);
    var restored = new ModelSession();
    AetherVk.Logic.SaveFile.SaveIndexMapping.Apply(dto, restored);
    Assert.True(restored.ShowJetAngles);
    s.Session.NotifyRestored();
    Assert.True(s.Vm.ShowJetAngles);
  }

  /// <summary>
  /// Selecting a jet, or moving its site, leaves every camera alone (it used to switch the
  /// viewports to comet orbiting facing the site). The tab only reaches cameras through the
  /// registry, which is never consulted.
  /// </summary>
  [Fact]
  public void SelectingOrEditingAJet_DoesNotTouchTheCameras()
  {
    var cameras = new Mock<ICameraServiceRegistry>(MockBehavior.Strict);
    var s = new TestSetup(cameras: cameras.Object);
    var (a, b) = TwoNativeJets(s);
    s.Vm.SelectedJet = a;
    s.Vm.SelectedJet = b;
    b.LatitudeDeg = 40;
    b.LongitudeDeg = -60;
    s.Vm.AddJetCommand.Execute(null);
    cameras.VerifyNoOtherCalls();
  }

  /// <summary>
  /// Savefile restore: the session is replaced from outside and <see cref="ModelSession.NotifyRestored"/>
  /// raised; the live tab re-reads the manual radius and forwards edits of the restored jets.
  /// </summary>
  [Fact]
  public void RestoredSession_IsPickedUp_ByTheLiveTab()
  {
    var s = new TestSetup();
    var jet = new JetViewModel(10, 20, 30, 2, 0.5f, new System.Numerics.Vector4(1, 1, 1, 1), true, 77u);
    s.Session.Jets.Clear();
    s.Session.Jets.Add(jet);
    s.Session.ManualNucleusRadiusKm = 3.5f;
    s.Session.NotifyRestored();
    Assert.Equal(1, jet.DisplayIndex);

    jet.NativePsId = 42;
    s.Runtime.Invocations.Clear();
    jet.LatitudeDeg = 11;
    s.Runtime.Verify(r => r.ModifyParticleSystem(42ul, It.IsAny<ParticleSystemModel>(),
      It.Is<ParticleSystemJet>(j => j.Seed == 77u), out It.Ref<ParticleSystemComputedProperties>.IsAny), Times.Once);
    // the restored manual radius is the one new jets start on
    ParticleSystemJet? added = null;
    ulong outId = 5;
    s.Runtime
      .Setup(r => r.AddParticleSystem(It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out outId))
      .Callback((ParticleSystemModel _, ParticleSystemJet j, out ulong id) => { added = j; id = 5; })
      .Returns(true);
    s.Runtime
      .Setup(r => r.AddFirstParticleSystem(It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out outId))
      .Callback((ParticleSystemModel _, ParticleSystemJet j, out ulong id) => { added = j; id = 5; })
      .Returns(new ParticleSystemComputedProperties(0.5f, 1.0f));
    s.Vm.AddJetCommand.Execute(null);
    Assert.Equal(3.5f, added!.NucleusRadiusKm);
  }

  /// Feedback item 10b: the nucleus radius moved to the Comet tab; the Model tab no longer has it.
  [Fact]
  public void ModelTab_NoLongerExposesTheManualRadius()
  {
    Assert.Null(typeof(IModelTabViewModel).GetProperty("ManualNucleusRadiusKm"));
    Assert.Null(typeof(IModelTabViewModel).GetProperty("ManualNucleusRadiusKmNullable"));
    Assert.Null(typeof(ModelTabViewModel).GetProperty("ManualNucleusRadiusKm"));
    Assert.NotNull(typeof(ICometTabViewModel).GetProperty("ManualNucleusRadiusKm"));
  }

  /// <summary>
  /// Dust view aids: tracers default off; the toggle sends the native flags with the flow bit
  /// always set (the flow runs while simulating and freezes on pause), is stored in the model
  /// session, and is visual only.
  /// </summary>
  [Fact]
  public void DustTracers_SendFlagsAndPersist_WithoutRestore()
  {
    var s = new TestSetup();
    s.Runtime.Invocations.Clear();
    Assert.False(s.Vm.DustTracers);
    Assert.False(s.Session.DustTracers);
    Assert.False(s.Vm.DustFlow);
    Assert.False(s.Session.DustFlow);

    s.Vm.DustTracers = true;
    s.Runtime.Verify(r => r.SetDustViewFlags(1u), Times.Once);
    Assert.True(s.Session.DustTracers);

    s.Vm.DustTracers = false;
    s.Runtime.Verify(r => r.SetDustViewFlags(0u), Times.Once);
    Assert.Equal(0u, ModelTabViewModel.DustViewFlags(false, false, false, false));
    s.Runtime.Verify(r => r.RestoreSnapshotSync(), Times.Never);
  }

  /// <summary>
  /// The flow pulses are a toggle, off by default (they drew the far-scale "rays" of the tail):
  /// on sends flag 2 with the tracers' bit, off clears it, and the session keeps the choice.
  /// </summary>
  [Fact]
  public void DustFlow_IsOffByDefault_ToggleSendsFlagAndPersists()
  {
    var s = new TestSetup();
    s.Runtime.Invocations.Clear();
    s.Vm.DustFlow = true;
    s.Runtime.Verify(r => r.SetDustViewFlags(2u), Times.Once);
    Assert.True(s.Session.DustFlow);
    s.Vm.DustTracers = true;
    s.Runtime.Verify(r => r.SetDustViewFlags(3u), Times.Once);
    s.Vm.DustFlow = false;
    s.Runtime.Verify(r => r.SetDustViewFlags(1u), Times.Once);
    Assert.False(s.Session.DustFlow);
    Assert.Equal(3u, ModelTabViewModel.DustViewFlags(true, true, false, false));
    s.Runtime.Verify(r => r.RestoreSnapshotSync(), Times.Never);
  }

  /// <summary>Every combination of the four dust view toggles maps to its native bits.</summary>
  [Fact]
  public void DustViewFlags_EveryCombination_MapsToItsBits()
  {
    for (uint bits = 0; bits < 16; bits++)
      Assert.Equal(bits, ModelTabViewModel.DustViewFlags(
        (bits & 1) != 0, (bits & 2) != 0, (bits & 4) != 0, (bits & 8) != 0));
  }

  /// <summary>
  /// Trajectory lines and the coma 1/ρ normalisation are off by default; each toggle sends its bit
  /// (4 and 8) together with the others, clearing it drops only that bit, and the session keeps it.
  /// </summary>
  [Fact]
  public void DustTrajectoriesAndRhoNorm_ToggleSendFlagsAndPersist_WithoutRestore()
  {
    var s = new TestSetup();
    s.Runtime.Invocations.Clear();
    Assert.False(s.Vm.DustTrajectories);
    Assert.False(s.Vm.DustRhoNorm);
    Assert.False(s.Session.DustTrajectories);
    Assert.False(s.Session.DustRhoNorm);

    s.Vm.DustTrajectories = true;
    s.Runtime.Verify(r => r.SetDustViewFlags(4u), Times.Once);
    Assert.True(s.Session.DustTrajectories);

    s.Vm.DustRhoNorm = true;
    s.Runtime.Verify(r => r.SetDustViewFlags(12u), Times.Once);
    Assert.True(s.Session.DustRhoNorm);

    s.Vm.DustTracers = true;
    s.Runtime.Verify(r => r.SetDustViewFlags(13u), Times.Once);

    s.Vm.DustTrajectories = false;
    s.Runtime.Verify(r => r.SetDustViewFlags(9u), Times.Once);
    Assert.False(s.Session.DustTrajectories);

    s.Vm.DustRhoNorm = false;
    s.Runtime.Verify(r => r.SetDustViewFlags(1u), Times.Once);
    Assert.False(s.Session.DustRhoNorm);
    s.Runtime.Verify(r => r.RestoreSnapshotSync(), Times.Never);
  }

  /// <summary>
  /// Dust exposure: 0 EV by default, goes to the runtime clamped by the UI to ±6 EV, is stored in
  /// the model session, is shown as "+1.25 EV", and is visual only.
  /// </summary>
  [Fact]
  public void DustExposureEv_SendsClampedValueAndPersists_WithoutRestore()
  {
    var s = new TestSetup();
    s.Runtime.Invocations.Clear();
    Assert.Equal(0.0, s.Vm.DustExposureEv);
    Assert.Equal(0.0, s.Session.DustExposureEv);

    s.Vm.DustExposureEv = 1.25;
    s.Runtime.Verify(r => r.SetDustExposureEv(1.25f), Times.Once);
    Assert.Equal(1.25, s.Session.DustExposureEv);
    Assert.Equal("+1.25 EV", s.Vm.DustExposureEvText);

    s.Vm.DustExposureEv = 50.0;
    Assert.Equal(6.0, s.Vm.DustExposureEv);
    Assert.Equal(6.0, s.Session.DustExposureEv);
    s.Runtime.Verify(r => r.SetDustExposureEv(6f), Times.Once);
    s.Runtime.Verify(r => r.SetDustExposureEv(50f), Times.Never);

    s.Vm.DustExposureEv = -50.0;
    Assert.Equal(-6.0, s.Vm.DustExposureEv);
    s.Runtime.Verify(r => r.SetDustExposureEv(-6f), Times.Once);
    Assert.Equal("-6.00 EV", s.Vm.DustExposureEvText);

    Assert.Equal(0.0, ModelTabViewModel.ClampDustExposureEv(double.NaN));
    Assert.Equal("0.00 EV", ModelTabViewModel.FormatDustExposureEv(0.0));
    s.Runtime.Verify(r => r.RestoreSnapshotSync(), Times.Never);
  }

  /// <summary>
  /// Flow time-lapse factor: goes to the runtime, is clamped to the native range [1, 10 000],
  /// is stored in the model session (restored on a session switch), and is visual only.
  /// </summary>
  [Fact]
  public void DustFlowSpeed_SendsClampedValueAndPersists_WithoutRestore()
  {
    var s = new TestSetup();
    s.Runtime.Invocations.Clear();
    Assert.Equal(1.0, s.Vm.DustFlowSpeed);

    s.Vm.DustFlowSpeed = 100.0;
    s.Runtime.Verify(r => r.SetDustFlowSpeed(100f), Times.Once);
    Assert.Equal(100.0, s.Session.DustFlowSpeed);

    s.Vm.DustFlowSpeed = 0.0;
    Assert.Equal(1.0, s.Vm.DustFlowSpeed);
    s.Runtime.Verify(r => r.SetDustFlowSpeed(1f), Times.Once);
    s.Vm.DustFlowSpeed = 1e9;
    Assert.Equal(10_000.0, s.Vm.DustFlowSpeed);
    s.Runtime.Verify(r => r.SetDustFlowSpeed(10_000f), Times.Once);
    Assert.Equal(1.0, ModelTabViewModel.ClampDustFlowSpeed(double.NaN));
    s.Runtime.Verify(r => r.RestoreSnapshotSync(), Times.Never);
  }

  /// <summary>
  /// "Dust visibility": the softening of the dust display stretch goes to the runtime, is clamped
  /// to [1e-7, 1], is stored in the model session, and is visual only (no snapshot restore,
  /// no jet re-push).
  /// </summary>
  [Fact]
  public void DustSoftening_IsSentClampedAndPersisted_WithoutRestore()
  {
    var s = new TestSetup();
    s.Runtime.Invocations.Clear();
    Assert.Equal(ModelTabDefaults.DustSoftening, s.Vm.DustSoftening);

    s.Vm.DustSoftening = 1e-3;
    s.Runtime.Verify(r => r.SetDustSoftening(1e-3f), Times.Once);
    Assert.Equal(1e-3, s.Session.DustSoftening);

    s.Vm.DustSoftening = 50.0;
    Assert.Equal(ModelTabDefaults.DustSofteningMax, s.Vm.DustSoftening);
    s.Runtime.Verify(r => r.SetDustSoftening((float)ModelTabDefaults.DustSofteningMax), Times.Once);
    s.Vm.DustSoftening = double.NaN;
    Assert.Equal(ModelTabDefaults.DustSoftening, s.Vm.DustSoftening);
    // a value persisted under the former absolute range (1e-7) clamps into the relative one
    s.Vm.DustSoftening = 1e-7;
    Assert.Equal(ModelTabDefaults.DustSofteningMin, s.Vm.DustSoftening);
    Assert.Equal(ModelTabDefaults.DustSofteningMin, s.Session.DustSoftening);

    s.Runtime.Verify(r => r.RestoreSnapshotSync(), Times.Never);
    s.Runtime.Verify(
      r => r.ModifyParticleSystem(It.IsAny<ulong>(), It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out It.Ref<ParticleSystemComputedProperties>.IsAny),
      Times.Never);
  }

  /// <summary>
  /// AddJetCommand must be disabled when no comet is committed, even if a nucleus
  /// radius is available. This prevents a silent FFI no-op.
  /// </summary>
  [Fact]
  public void AddJetCommand_DisabledWhenNoCometCommitted()
  {
    var s = new TestSetup(nucleusRadiusKm: 5.0f, cometCommitted: false);
    Assert.False(s.Vm.AddJetCommand.CanExecute(null),
      "AddJetCommand must be disabled when no comet is committed");
  }

  /// <summary>
  /// AddJetCommand becomes enabled once IsCometCommitted transitions to true
  /// (and a nucleus radius is available).
  /// </summary>
  [Fact]
  public void AddJetCommand_EnabledAfterCometCommitted()
  {
    var s = new TestSetup(nucleusRadiusKm: 5.0f, cometCommitted: false);
    Assert.False(s.Vm.AddJetCommand.CanExecute(null));

    s.Vm.IsCometCommitted = true;

    Assert.True(s.Vm.AddJetCommand.CanExecute(null),
      "AddJetCommand must be enabled once IsCometCommitted becomes true");
  }

  /// <summary>
  /// AddJet must call AddFirstParticleSystem exactly once for the first jet.
  /// </summary>
  [Fact]
  public void AddJet_FirstJet_CallsAddFirstParticleSystem()
  {
    var s = new TestSetup();
    ulong outId = 42;
    s.Runtime
      .Setup(r => r.AddFirstParticleSystem(
        It.IsAny<ParticleSystemModel>(),
        It.IsAny<ParticleSystemJet>(),
        out outId))
      .Returns(new ParticleSystemComputedProperties(0.5f, 1.2f));

    s.Vm.AddJetCommand.Execute(null);

    s.Runtime.Verify(
      r => r.AddFirstParticleSystem(
        It.IsAny<ParticleSystemModel>(),
        It.IsAny<ParticleSystemJet>(),
        out outId),
      Times.Once);
  }

  /// <summary>
  /// AddJet for the second jet must call AddParticleSystem (not AddFirstParticleSystem).
  /// </summary>
  [Fact]
  public void AddJet_SecondJet_CallsAddParticleSystem()
  {
    var s = new TestSetup();
    ulong firstId = 10;
    ulong secondId = 11;
    s.Runtime
      .Setup(r => r.AddFirstParticleSystem(
        It.IsAny<ParticleSystemModel>(),
        It.IsAny<ParticleSystemJet>(),
        out firstId))
      .Returns(new ParticleSystemComputedProperties(0.5f, 1.2f));
    s.Runtime
      .Setup(r => r.AddParticleSystem(
        It.IsAny<ParticleSystemModel>(),
        It.IsAny<ParticleSystemJet>(),
        out secondId))
      .Returns(true);

    s.Vm.AddJetCommand.Execute(null); // first
    s.Vm.AddJetCommand.Execute(null); // second

    s.Runtime.Verify(
      r => r.AddParticleSystem(
        It.IsAny<ParticleSystemModel>(),
        It.IsAny<ParticleSystemJet>(),
        out secondId),
      Times.Once);
  }

  /// <summary>
  /// The model (common) parameters sourced from ModelSession must be passed verbatim to the native call.
  /// </summary>
  [Fact]
  public void AddJet_PassesModelSessionCommonParamsToNative()
  {
    var s = new TestSetup();
    s.Session.DiametreUm = 42.0f;
    s.Session.DensityGCm3 = 1.3f;

    ParticleSystemModel? captured = null;
    ulong outId = 1;
    s.Runtime
      .Setup(r => r.AddFirstParticleSystem(
        It.IsAny<ParticleSystemModel>(),
        It.IsAny<ParticleSystemJet>(),
        out outId))
      .Callback((ParticleSystemModel model, ParticleSystemJet jet, out ulong id) =>
      {
        captured = model;
        id = 1;
      })
      .Returns(new ParticleSystemComputedProperties(0.5f, 1.0f));

    s.Vm.AddJetCommand.Execute(null);

    Assert.NotNull(captured);
    Assert.Equal(42.0f, captured!.DiametreUm);
    Assert.Equal(1.3f, captured.DensityGCm3);
  }

  /// <summary>
  /// RemoveJet must call RemoveParticleSystem with the correct NativePsId.
  /// </summary>
  [Fact]
  public void RemoveJet_CallsRemoveParticleSystemWithCorrectId()
  {
    var s = new TestSetup();
    ulong assignedId = 99;
    s.Runtime
      .Setup(r => r.AddFirstParticleSystem(
        It.IsAny<ParticleSystemModel>(),
        It.IsAny<ParticleSystemJet>(),
        out assignedId))
      .Returns(new ParticleSystemComputedProperties(0.5f, 1.0f));
    s.Runtime.Setup(r => r.RemoveParticleSystem(It.IsAny<ulong>())).Returns(true);

    s.Vm.AddJetCommand.Execute(null);
    var jet = s.Vm.Jets!.Single();

    s.Vm.RemoveJetCommand.Execute(jet);

    s.Runtime.Verify(r => r.RemoveParticleSystem(assignedId), Times.Once);
    Assert.Empty(s.Vm.Jets!);
  }

  /// <summary>
  /// The NucleusRadiusKm passed in the jet struct must equal EffectiveNucleusRadiusKm.
  /// </summary>
  [Fact]
  public void AddJet_PassesNucleusRadiusKmToNative()
  {
    var s = new TestSetup(nucleusRadiusKm: 7.5f);

    ParticleSystemJet? capturedJet = null;
    ulong outId = 2;
    s.Runtime
      .Setup(r => r.AddFirstParticleSystem(
        It.IsAny<ParticleSystemModel>(),
        It.IsAny<ParticleSystemJet>(),
        out outId))
      .Callback((ParticleSystemModel model, ParticleSystemJet jet, out ulong id) =>
      {
        capturedJet = jet;
        id = 2;
      })
      .Returns(new ParticleSystemComputedProperties(0.5f, 1.0f));

    s.Vm.AddJetCommand.Execute(null);

    Assert.NotNull(capturedJet);
    Assert.Equal(7.5f, capturedJet!.NucleusRadiusKm);
  }

  /// <summary>
  /// ModelSession default values must match the documented physics defaults.
  /// </summary>
  [Fact]
  public void ModelSession_DefaultValues_ArePhysicallyReasonable()
  {
    var session = new ModelSession();
    Assert.Equal(100f, session.DiametreUm);
    Assert.Equal(0.8f, session.DensityGCm3, 3);
    Assert.Equal(1.0f, session.ScatteringEfficiency);
    Assert.Equal(0.04f, session.GeometricAlbedo);
    Assert.Equal(3.5f, session.SizeQ);
    Assert.Equal(10f, session.SizeRange);
    Assert.Equal(2f, session.DustRate1AuKgs);
    Assert.Equal(2.0f, session.DustRatePower);
    Assert.Equal(15.0f, session.DustRateCutoffAu);
    Assert.Equal(2000f, session.DustRateMaxKgs);
  }

  /// <summary>
  /// ModelSession common properties must raise PropertyChanged when set,
  /// so the ViewModel can re-plot the defaults and push the model-wide Q_d law to every jet.
  /// </summary>
  [Fact]
  public void ModelSession_CommonProperties_RaisePropertyChanged()
  {
    var session = new ModelSession();
    var raised = new System.Collections.Generic.List<string?>();
    session.PropertyChanged += (_, e) => raised.Add(e.PropertyName);

    session.DiametreUm = 25.0f;
    session.DensityGCm3 = 1.1f;
    session.ScatteringEfficiency = 1.5f;
    session.GeometricAlbedo = 0.06f;
    session.SizeQ = 3.0f;
    session.SizeRange = 20f;
    session.DustRate1AuKgs = 10f;
    session.DustRatePower = 3.0f;
    session.DustRateCutoffAu = 6.0f;
    session.DustRateMaxKgs = 4000f;

    Assert.Contains(nameof(ModelSession.DiametreUm), raised);
    Assert.Contains(nameof(ModelSession.DensityGCm3), raised);
    Assert.Contains(nameof(ModelSession.ScatteringEfficiency), raised);
    Assert.Contains(nameof(ModelSession.GeometricAlbedo), raised);
    Assert.Contains(nameof(ModelSession.SizeQ), raised);
    Assert.Contains(nameof(ModelSession.SizeRange), raised);
    Assert.Contains(nameof(ModelSession.DustRate1AuKgs), raised);
    Assert.Contains(nameof(ModelSession.DustRatePower), raised);
    Assert.Contains(nameof(ModelSession.DustRateCutoffAu), raised);
    Assert.Contains(nameof(ModelSession.DustRateMaxKgs), raised);
  }

  // ── Compound CanAddJet gate ────────────────────────────────────────────────

  /// <summary>
  /// AddJetCommand must be disabled when the comet IS committed but no nucleus
  /// radius is available. Both conditions must be satisfied simultaneously.
  /// </summary>
  [Fact]
  public void AddJetCommand_DisabledWhenCometCommittedButNoRadius()
  {
    // cometCommitted=true, nucleusRadiusKm=0 → CanAddJet() must return false
    var s = new TestSetup(nucleusRadiusKm: 0f, cometCommitted: true);
    Assert.False(s.Vm.AddJetCommand.CanExecute(null),
      "AddJetCommand must be disabled when radius is 0 even if comet is committed");
  }

  // ── PushModelToAllJets reactive pipeline ──────────────────────────────────

  /// <summary>
  /// When a model-wide property (the Q_d law's shape) changes, PushModelToAllJets must call
  /// ModifyParticleSystem for every jet with a non-zero NativePsId, each with its own grain.
  /// A grain default (diameter …) changes no existing jet. Verifies the full reactive pipeline:
  /// PropertyChanged → Throttle → PushModelToAllJets → FFI. ImmediateScheduler (used in TestSetup)
  /// collapses the Throttle delay to zero.
  /// </summary>
  [Fact]
  public void ModelSessionPropertyChange_AfterThrottle_CallsModifyParticleSystem()
  {
    var s = new TestSetup(nucleusRadiusKm: 5.0f, cometCommitted: true);

    // Add one jet
    ulong jetId = 77;
    s.Runtime
      .Setup(r => r.AddFirstParticleSystem(
        It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out jetId))
      .Returns(new ParticleSystemComputedProperties(0.5f, 1.0f));

    // Track ModifyParticleSystem invocations manually (avoids Moq out-param verify issues)
    var modifyCalls = new List<(ulong id, ParticleSystemModel model)>();
    s.Runtime
      .Setup(r => r.ModifyParticleSystem(
        It.IsAny<ulong>(), It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(),
        out It.Ref<ParticleSystemComputedProperties>.IsAny))
      .Callback(new ModifyParticleSystemCallback((ulong id, ParticleSystemModel model,
          ParticleSystemJet jet, out ParticleSystemComputedProperties comp) =>
        {
          modifyCalls.Add((id, model));
          comp = new ParticleSystemComputedProperties(0.1f, 0.2f);
        }))
      .Returns(true);

    s.Vm.AddJetCommand.Execute(null);

    // No modify calls before session property changes
    Assert.Empty(modifyCalls);

    // a default for new jets: the existing jet keeps its grain, nothing is pushed
    s.Session.DiametreUm = 50.0f;
    Assert.Empty(modifyCalls);
    Assert.Equal(100f, s.Vm.Jets!.Single().DiametreUm);

    // Mutate a model-wide property — with ImmediateScheduler the Throttle fires synchronously
    s.Session.DustRatePower = 3.0f;

    // Pipeline must have fired: PropertyChanged → Throttle → PushModelToAllJets → ModifyParticleSystem
    Assert.Single(modifyCalls);
    Assert.Equal(jetId, modifyCalls[0].id);
    Assert.Equal(3.0f, modifyCalls[0].model.DustRatePower);
    Assert.Equal(100f, modifyCalls[0].model.DiametreUm);
  }

  // ── Per-jet grain and activity ─────────────────────────────────────────────

  /// <summary>Adds two native jets (ids 21, 22) and records every ModifyParticleSystem call.</summary>
  private static List<(ulong id, ParticleSystemModel model)> AddTwoJetsRecordingModifies(TestSetup s)
  {
    ulong firstId = 21, secondId = 22;
    s.Runtime
      .Setup(r => r.AddFirstParticleSystem(It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out firstId))
      .Returns(new ParticleSystemComputedProperties(0.5f, 1.0f));
    s.Runtime
      .Setup(r => r.AddParticleSystem(It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out secondId))
      .Returns(true);
    var modifyCalls = new List<(ulong id, ParticleSystemModel model)>();
    s.Runtime
      .Setup(r => r.ModifyParticleSystem(
        It.IsAny<ulong>(), It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(),
        out It.Ref<ParticleSystemComputedProperties>.IsAny))
      .Callback(new ModifyParticleSystemCallback((ulong id, ParticleSystemModel model,
          ParticleSystemJet jet, out ParticleSystemComputedProperties comp) =>
        {
          modifyCalls.Add((id, model));
          // β ∝ 1/diameter, like the engine (enough to tell the jets apart)
          comp = new ParticleSystemComputedProperties(10f / model.DiametreUm, model.DustRate1AuKgs);
        }))
      .Returns(true);
    s.Vm.AddJetCommand.Execute(null);
    s.Vm.AddJetCommand.Execute(null);
    return modifyCalls;
  }

  /// <summary>
  /// Editing one jet's grain diameter modifies that jet only: one native modify of its system with
  /// the new diameter, the sibling's grain (view model and native) untouched.
  /// </summary>
  [Fact]
  public void EditingOneJetDiameter_ModifiesOnlyThatJet()
  {
    var s = new TestSetup();
    var modifyCalls = AddTwoJetsRecordingModifies(s);
    var (first, second) = (s.Vm.Jets![0], s.Vm.Jets[1]);
    Assert.Equal((21ul, 22ul), (first.NativePsId, second.NativePsId));
    // the later jet read its own β and rate back (AddParticleSystem returns none)
    Assert.Equal(10f / 100f, second.Beta, 6);
    modifyCalls.Clear();

    first.DiametreUm = 20f;

    var call = Assert.Single(modifyCalls);
    Assert.Equal(21ul, call.id);
    Assert.Equal(20f, call.model.DiametreUm);
    Assert.Equal(0.5f, first.Beta, 6);
    Assert.Equal(100f, second.DiametreUm);
    Assert.Equal(10f / 100f, second.Beta, 6);
  }

  /// <summary>
  /// A new jet starts from the model session's grain and activity defaults (view model and native
  /// record), and a later change of the defaults leaves it as it is.
  /// </summary>
  [Fact]
  public void NewJet_TakesTheModelDefaults()
  {
    var s = new TestSetup();
    s.Session.DiametreUm = 42f;
    s.Session.DensityGCm3 = 1.3f;
    s.Session.ScatteringEfficiency = 1.4f;
    s.Session.GeometricAlbedo = 0.06f;
    s.Session.SizeQ = 3.0f;
    s.Session.SizeRange = 30f;
    s.Session.DustRate1AuKgs = 7f;
    s.Session.DustRatePower = 2.5f;
    ParticleSystemModel? captured = null;
    ulong outId = 5;
    s.Runtime
      .Setup(r => r.AddFirstParticleSystem(It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out outId))
      .Callback((ParticleSystemModel model, ParticleSystemJet jet, out ulong id) =>
      {
        captured = model;
        id = 5;
      })
      .Returns(new ParticleSystemComputedProperties(0.5f, 1.0f));

    s.Vm.AddJetCommand.Execute(null);

    var jet = s.Vm.Jets!.Single();
    // q is per jet only: a new jet starts at 3.5 whatever the session holds (an older save's value)
    var expected = new AetherVk.Logic.Models.JetGrain(42f, 1.3f, 1.4f, 0.06f, 3.5f, 30f, 7f);
    Assert.Equal(expected, jet.Grain);
    Assert.Equal(expected with { SizeQ = 3.0f }, s.Session.DefaultGrain);
    Assert.Equal(new ParticleSystemModel(42f, 1.3f, 1.4f, 0.06f, 3.5f, 30f, 7f, 2.5f, 15f, 2000f), captured);

    s.Session.DiametreUm = 7f;
    s.Session.DustRate1AuKgs = 9f;
    Assert.Equal(expected, jet.Grain);
  }

  /// <summary>"Copy to all jets" gives the siblings the selected jet's grain and Q_d₀, nothing else.</summary>
  [Fact]
  public void CopyGrainToAllJets_CopiesTheGrainOnly()
  {
    var s = new TestSetup();
    var modifyCalls = AddTwoJetsRecordingModifies(s);
    var (first, second) = (s.Vm.Jets![0], s.Vm.Jets[1]);
    var site = (second.LatitudeDeg, second.LongitudeDeg, second.StartVelocityMeanMs, second.StreamColor);
    first.Grain = new AetherVk.Logic.Models.JetGrain(250f, 0.8f, 1.2f, 0.05f, 3.2f, 20f, 900f);
    s.Vm.SelectedJet = first;
    modifyCalls.Clear();

    s.Vm.CopyGrainToAllJetsCommand.Execute(null);

    Assert.Equal(first.Grain, second.Grain);
    Assert.Equal(site, (second.LatitudeDeg, second.LongitudeDeg, second.StartVelocityMeanMs, second.StreamColor));
    Assert.Contains(modifyCalls, c => c.id == 22ul && c.model.DiametreUm == 250f && c.model.DustRate1AuKgs == 900f);
    Assert.DoesNotContain(modifyCalls, c => c.id == 21ul);
  }

  /// <summary>The Σ read-out adds the jets' Q_d₀.</summary>
  [Fact]
  public void TotalActivityText_SumsTheJets()
  {
    var s = new TestSetup();
    AddTwoJetsRecordingModifies(s);
    s.Vm.Jets![0].DustRate1AuKgs = 1.5f;
    s.Vm.Jets[1].DustRate1AuKgs = 2.5f;
    Assert.Contains("Σ 2 jets at 1 AU: Σ Q_d 4 kg/s", s.Vm.TotalActivityText);
  }

  /// <summary>No "Afρ" survives in the strings the Model tab exposes (reviewer request: Q_d only).</summary>
  [Fact]
  public void ExposedStrings_MentionNoAfrho()
  {
    var s = new TestSetup();
    AddTwoJetsRecordingModifies(s);
    s.Vm.SelectedJet = s.Vm.Jets![0];
    foreach (var text in new[] { s.Vm.TotalActivityText, s.Vm.DustRateNowText,
               s.Vm.DustRatePlot?.YLabel ?? "", s.Vm.JetDustRatePlot?.YLabel ?? "",
               s.Vm.DustRatePlot?.Series[0].Label ?? "", s.Vm.JetDustRatePlot?.Series[0].Label ?? "" })
    {
      Assert.DoesNotContain("Afρ", text);
      Assert.DoesNotContain("afrho", text, System.StringComparison.OrdinalIgnoreCase);
    }
  }

  // ── RemoveJet edge cases ───────────────────────────────────────────────────

  /// <summary>
  /// RemoveJet must NOT call RemoveParticleSystem when the jet has NativePsId == 0
  /// (i.e. the add call failed and no native entity was created).
  /// </summary>
  [Fact]
  public void RemoveJet_WithZeroPsId_DoesNotCallRemoveParticleSystem()
  {
    var s = new TestSetup();
    // Add a jet that "fails" — mock returns psId=0
    ulong failedId = 0;
    s.Runtime
      .Setup(r => r.AddFirstParticleSystem(
        It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out failedId))
      .Returns(new ParticleSystemComputedProperties(0f, 0f));

    s.Vm.AddJetCommand.Execute(null);
    var jet = s.Vm.Jets!.Single();
    Assert.Equal(0UL, jet.NativePsId);

    s.Vm.RemoveJetCommand.Execute(jet);

    s.Runtime.Verify(r => r.RemoveParticleSystem(It.IsAny<ulong>()), Times.Never,
      "RemoveParticleSystem must not be called when NativePsId is 0");
    Assert.Empty(s.Vm.Jets!);
  }

  /// <summary>
  /// After RemoveJet the removed jet's ID must not receive any further
  /// ModifyParticleSystem calls (e.g. from PushModelToAllJets), because it is
  /// no longer in session.Jets.
  /// </summary>
  [Fact]
  public void RemoveJet_RemovedJetDoesNotReceiveModify()
  {
    var s = new TestSetup();
    ulong jetId = 55;
    s.Runtime
      .Setup(r => r.AddFirstParticleSystem(
        It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out jetId))
      .Returns(new ParticleSystemComputedProperties(0.5f, 1.0f));
    s.Runtime.Setup(r => r.RemoveParticleSystem(It.IsAny<ulong>())).Returns(true);

    s.Vm.AddJetCommand.Execute(null);
    var jet = s.Vm.Jets!.Single();
    s.Vm.RemoveJetCommand.Execute(jet);

    // PushModelToAllJets iterates session.Jets — now empty — so no ModifyParticleSystem calls
    s.Runtime.Verify(r => r.ModifyParticleSystem(
        jetId, It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(),
        out It.Ref<ParticleSystemComputedProperties>.IsAny),
      Times.Never,
      "Removed jet must not receive ModifyParticleSystem calls");
  }
}
