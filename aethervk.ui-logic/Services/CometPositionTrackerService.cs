
using System;
using System.Numerics;
using System.Reactive.Linq;
using System.Reactive.Subjects;

namespace AetherVk.Logic.Services;

/// <summary>
/// Internal state machine for <see cref="CometPositionTrackerService"/>.
/// </summary>
internal enum CometTrackerState
{
  /// No SPK loaded. Comet is at a default, hardcoded position (1 AU, fixed orientation).
  DefaultComet,
  /// SPK loaded and timeline is valid. First position has been fetched for <c>start_epoch</c>.
  TimelineValidated,
  /// Simulation is running. Service forwards position updates from <c>SIMULATION_CALLBACK</c>.
  SimulationRunning,
  /// Simulation was stopped and the comet subtree was removed from the scene.
  /// Observable emits <c>null</c>. Resets to <see cref="DefaultComet"/> when new data arrives.
  CometRemoved,
}

/// <summary>
/// Tracks the position of the comet nucleus entity in the simulation scene.
///
/// <para><b>State machine:</b></para>
/// <list type="bullet">
///   <item><c>DefaultComet</c> — initial state; default 1-AU position emitted.</item>
///   <item><c>TimelineValidated</c> — SPK loaded and timeline confirmed by <see cref="TimelineService"/>;
///     first position fetched once via a sync FFI call (TODO: pending Rust-side addition of
///     <c>avkSimulationContext_getCometPositionAtEpoch</c>).</item>
///   <item><c>SimulationRunning</c> — comet position forwarded live from <c>SIMULATION_CALLBACK</c>.</item>
///   <item><c>CometRemoved</c> — sim stopped and comet subtree removed; <c>null</c> emitted.
///     Reverts to <c>DefaultComet</c> when comet data is detached and default subtree re-inserted.</item>
/// </list>
///
/// <para>The observable is always <c>ObserveOn(MainThread)</c> since position is used for UI display.
/// <see cref="CameraService"/> subscribes to the raw subject directly (no extra hop).</para>
///
/// - part of the "Companion Runtime Service" group
/// </summary>
/// <seealso cref="CameraService" />
/// <seealso cref="TimelineService" />
/// <seealso cref="ImportedModelsTrackerService" />
public sealed class CometPositionTrackerService : IDisposable
{
  private readonly INativeRuntimeService _runtimeService;
  private readonly ISchedulerProvider _schedulerProvider;

  // null = no comet (or sim stopped + comet removed)
  private readonly BehaviorSubject<Vector3?> _positionSubject = new(null);

  // f64 shadow of the comet position — kept in sync with _positionSubject but never
  // truncated to f32. Used by CameraService to compute the orbit-offset addition without
  // catastrophic cancellation at large heliocentric distances.
  private (double X, double Y, double Z)? _lastKnownCometPositionF64;

  private CometTrackerState _state = CometTrackerState.DefaultComet;
  private IDisposable? _simListenerToken;
  private IDisposable? _timelineSubscription;
  private IDisposable? _cometSnapshotListenerToken;

  public CometPositionTrackerService(
    INativeRuntimeService runtimeService,
    ISchedulerProvider schedulerProvider,
    TimelineService timelineService)
  {
    _runtimeService = runtimeService;
    _schedulerProvider = schedulerProvider;

    // Emit the synthetic default position at startup (before any SPK is loaded)
    EmitDefaultPosition();

    // Subscribe to timeline validity — transitions DefaultComet → TimelineValidated
    _timelineSubscription = timelineService.IsTimelineValid
      .Subscribe(isValid =>
      {
        if (isValid)
          OnTimelineValidated(timelineService);
        else
          OnTimelineInvalidated();
      });

    // Subscribe to the post-commit position snapshot so the position is available
    // immediately after BuildCometTrajectory completes, even when the simulation is paused.
    _cometSnapshotListenerToken = runtimeService.RegisterExternalStateListener(
      ExternalStateType.CometPositionSnapshot,
      HandleCometPositionSnapshotCallback);
  }

  // ── Observables ────────────────────────────────────────────────────────────

  /// <summary>
  /// Current comet position in simulation units (AU), or <c>null</c> when the comet is
  /// absent from the scene. Observed on the main-thread scheduler.
  /// </summary>
  public IObservable<Vector3?> CometPosition =>
    _positionSubject.ObserveOn(_schedulerProvider.MainThread);

  /// <summary>
  /// Raw subject — for internal consumers (e.g. <see cref="CameraService"/>) that must
  /// react on the callback thread without an extra scheduler hop.
  /// </summary>
  internal IObservable<Vector3?> CometPositionRaw => _positionSubject;

  /// <summary>
  /// Synchronous read of the last known comet position.
  /// Used by <see cref="CameraService"/> to apply orbit zoom immediately when the
  /// simulation is paused (i.e. no pending <c>SIMULATION_CALLBACK</c> tick to trigger
  /// <c>SnapCameraToOrbit</c>).
  /// </summary>
  internal Vector3? LastKnownCometPosition => _positionSubject.Value;

  /// <summary>
  /// Synchronous read of the last known comet position in full double precision (AU).
  /// Kept in sync with <see cref="LastKnownCometPosition"/> but without the f64→f32 truncation.
  /// Used by <see cref="CameraService"/> to compute <c>cometPos + orbitOffset</c> in f64
  /// so the result survives f32 addition at large heliocentric distances (catastrophic cancellation
  /// occurs when the offset is small relative to the position magnitude in f32).
  /// </summary>
  internal (double X, double Y, double Z)? LastKnownCometPositionF64 => _lastKnownCometPositionF64;


  // ── State transitions ──────────────────────────────────────────────────────

  private void EmitDefaultPosition()
  {
    // Default comet: 1 AU from the sun in the +X direction (arbitrary, matches Rust initial scene)
    _positionSubject.OnNext(new Vector3(1.0f, 0f, 0f));
    _state = CometTrackerState.DefaultComet;
  }

  private void OnTimelineValidated(TimelineService timelineService)
  {
    if (_state is CometTrackerState.SimulationRunning) return; // already receiving live updates

    _state = CometTrackerState.TimelineValidated;

    // TODO (Rust): call avkSimulationContext_getCometPositionAtEpoch(startEpoch) here.
    // For now we keep the last emitted position (default or previous).
    // Once the FFI exists:
    //   if (_runtimeService.CometEntityId is ulong cometId)
    //   {
    //     var range = timelineService.ValidatedTimeRange.FirstOrDefault(); // sync peek
    //     FetchFirstCometPosition(cometId, range?.StartCenturies, range?.StartNs);
    //   }

    // Register simulation listener for live position updates (also used in SimulationRunning)
    RegisterCometSimListener();
  }

  private void OnTimelineInvalidated()
  {
    // Timeline became invalid (e.g. SPK detached) — emit default position
    DeregisterCometSimListener();
    EmitDefaultPosition();
  }

  /// <summary>
  /// Called by <see cref="CameraService"/> (or any orchestrator) when the simulation starts.
  /// </summary>
  internal void OnSimulationStarted()
  {
    _state = CometTrackerState.SimulationRunning;
    RegisterCometSimListener(); // idempotent — only registers once
  }

  /// <summary>
  /// Called when the simulation stops. If the comet subtree was removed, emits <c>null</c>.
  /// </summary>
  internal void OnSimulationStopped(bool cometSubtreeRemoved)
  {
    _state = cometSubtreeRemoved ? CometTrackerState.CometRemoved : CometTrackerState.TimelineValidated;
    if (cometSubtreeRemoved)
    {
      _positionSubject.OnNext(null);
      // When a new comet is configured the service will transition back via OnTimelineValidated
    }
    // If not removed, keep the last known position (sim paused = comet frozen)
  }

  // ── Simulation listener management ────────────────────────────────────────

  private void RegisterCometSimListener()
  {
    if (_simListenerToken is not null)
    {
      Console.WriteLine("[CometPositionTrackerService] RegisterCometSimListener: already registered, skipping.");
      return;
    }

    ulong? cometEntityId = _runtimeService.CometEntityId;
    Console.WriteLine($"[CometPositionTrackerService] RegisterCometSimListener: CometEntityId={cometEntityId?.ToString() ?? "null"} state={_state}");
    if (cometEntityId is null)
    {
      // TODO (Rust): revisit once CometEntityId is populated from reconfigureComet out-param
      Console.WriteLine("[CometPositionTrackerService] RegisterCometSimListener: SKIPPED — CometEntityId is null. Position callbacks will not fire.");
      return;
    }

    Console.WriteLine($"[CometPositionTrackerService] RegisterCometSimListener: Registering for entity {cometEntityId.Value} comp={ComponentForeignId.HighResTransform}");
    _simListenerToken = _runtimeService.RegisterSimulationListener(
      cometEntityId.Value,
      ComponentForeignId.HighResTransform,
      HandleCometPositionCallback);
  }

  private void DeregisterCometSimListener()
  {
    _simListenerToken?.Dispose();
    _simListenerToken = null;
  }

  // ── Internal callback handling ─────────────────────────────────────────────

  // Invoked on the native callback thread — must not block, must not throw.
  private unsafe void HandleCometPositionCallback(nint dataPtr)
  {
    var dto = *(HighResTransformDTO*)dataPtr;
    Console.WriteLine($"[CometPositionTrackerService] SIMULATION_CALLBACK comet pos: ({dto.PosX:F6}, {dto.PosY:F6}, {dto.PosZ:F6}) AU");
    // Store f64 before truncating to f32 for the subject
    _lastKnownCometPositionF64 = (dto.PosX, dto.PosY, dto.PosZ);
    _positionSubject.OnNext(new Vector3((float)dto.PosX, (float)dto.PosY, (float)dto.PosZ));
  }

  // Invoked on the native callback thread when BuildCometTrajectory emits
  // ExternalState::CometPositionSnapshot. Updates _positionSubject immediately so
  // CameraService can enter CometOrbiting mode at the correct position even when
  // the simulation is not yet running.
  private unsafe void HandleCometPositionSnapshotCallback(nint dataPtr)
  {
    var dto = *(CCometPositionSnapshotDTO*)dataPtr;
    // Store f64 before truncating to f32 for the subject
    _lastKnownCometPositionF64 = (dto.PosX, dto.PosY, dto.PosZ);
    var pos = new Vector3((float)dto.PosX, (float)dto.PosY, (float)dto.PosZ);
    Console.WriteLine(
      $"[CometPositionTrackerService] CometPositionSnapshot received: SPK={dto.SpkId} "
      + $"pos=({dto.PosX:F6}, {dto.PosY:F6}, {dto.PosZ:F6}) AU → f32=({pos.X:F4}, {pos.Y:F4}, {pos.Z:F4})"
    );
    _positionSubject.OnNext(pos);
  }


  // ── IDisposable ────────────────────────────────────────────────────────────

  public void Dispose()
  {
    _timelineSubscription?.Dispose();
    _cometSnapshotListenerToken?.Dispose();
    DeregisterCometSimListener();
    _positionSubject.Dispose();
  }
}
