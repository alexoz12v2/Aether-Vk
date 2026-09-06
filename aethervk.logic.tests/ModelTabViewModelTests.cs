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
/// common-parameter propagation. Native FFI paths are bypassed via a mocked
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

    public TestSetup(float nucleusRadiusKm = 5.0f, bool cometCommitted = true)
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

      Session = new ModelSession();
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

      Vm = new ModelTabViewModel(
        translationService.Object,
        schedulers.Object,
        sessionService.Object,
        cometSessionService.Object,
        cometConfig,
        Runtime.Object,
        dispatcher.Object,
        cometMessenger.Object,
        new Mock<IPlatformWindowService>().Object
      ); // Most tests need a committed comet — simulate commitment via the observable property.
      if (cometCommitted)
        Vm.IsCometCommitted = true;

      Vm.ManualNucleusRadiusKm = nucleusRadiusKm;
    }
  }

  // ── tests ─────────────────────────────────────────────────────────────────────

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
    Assert.Equal(0.30f, session.MassVariabilityPerc);
    Assert.Equal(100f, session.DiametreUm);
    Assert.Equal(0.533f, session.DensityGCm3, 3);
    Assert.Equal(1.0f, session.ScatteringEfficiency);
    Assert.Equal(100f, session.Afrho0Cm);
    Assert.Equal(2.0f, session.AfrhoPower);
    Assert.Equal(5.0f, session.AfrhoCutoffAu);
    Assert.Equal(100_000f, session.AfrhoMaxValueCm);
  }

  /// <summary>
  /// ModelSession common properties must raise PropertyChanged when set,
  /// so the ViewModel can detect model-level edits and propagate them to siblings.
  /// </summary>
  [Fact]
  public void ModelSession_CommonProperties_RaisePropertyChanged()
  {
    var session = new ModelSession();
    var raised = new System.Collections.Generic.List<string?>();
    session.PropertyChanged += (_, e) => raised.Add(e.PropertyName);

    session.DiametreUm = 25.0f;
    session.DensityGCm3 = 1.1f;
    session.MassVariabilityPerc = 0.4f;
    session.ScatteringEfficiency = 1.5f;
    session.Afrho0Cm = 1000f;
    session.AfrhoPower = 3.0f;
    session.AfrhoCutoffAu = 6.0f;
    session.AfrhoMaxValueCm = 200_000f;

    Assert.Contains(nameof(ModelSession.DiametreUm), raised);
    Assert.Contains(nameof(ModelSession.DensityGCm3), raised);
    Assert.Contains(nameof(ModelSession.MassVariabilityPerc), raised);
    Assert.Contains(nameof(ModelSession.ScatteringEfficiency), raised);
    Assert.Contains(nameof(ModelSession.Afrho0Cm), raised);
    Assert.Contains(nameof(ModelSession.AfrhoPower), raised);
    Assert.Contains(nameof(ModelSession.AfrhoCutoffAu), raised);
    Assert.Contains(nameof(ModelSession.AfrhoMaxValueCm), raised);
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
  /// When a ModelSession common property changes, PushModelToAllJets must call
  /// ModifyParticleSystem for every jet with a non-zero NativePsId.
  /// Verifies the full reactive pipeline: PropertyChanged → Throttle → PushModelToAllJets → FFI.
  /// ImmediateScheduler (used in TestSetup) collapses the Throttle delay to zero.
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

    // Mutate a common property — with ImmediateScheduler the Throttle fires synchronously
    s.Session.DiametreUm = 50.0f;

    // Pipeline must have fired: PropertyChanged → Throttle → PushModelToAllJets → ModifyParticleSystem
    Assert.Single(modifyCalls);
    Assert.Equal(jetId, modifyCalls[0].id);
    Assert.Equal(50.0f, modifyCalls[0].model.DiametreUm);
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
    ParticleSystemComputedProperties computed = default;
    s.Runtime.Verify(r => r.ModifyParticleSystem(
        jetId, It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(),
        out computed),
      Times.Never,
      "Removed jet must not receive ModifyParticleSystem calls");
  }
}
