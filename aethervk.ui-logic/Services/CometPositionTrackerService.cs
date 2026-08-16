
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

  private CometTrackerState _state = CometTrackerState.DefaultComet;
  private IDisposable? _simListenerToken;
  private IDisposable? _timelineSubscription;

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
    if (_simListenerToken is not null) return; // already registered

    ulong? cometEntityId = _runtimeService.CometEntityId;
    if (cometEntityId is null)
    {
      // TODO (Rust): revisit once CometEntityId is populated from reconfigureComet out-param
      return;
    }

    _simListenerToken = _runtimeService.RegisterSimulationListener(
      cometEntityId.Value,
      ComponentForeignId.CometPosition,
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
    var dto = *(CometPositionDTO*)dataPtr;
    // Cast f64 → f32 for Vector3 (precision sufficient for UI coordinates)
    _positionSubject.OnNext(new Vector3((float)dto.X, (float)dto.Y, (float)dto.Z));
  }

  // ── IDisposable ────────────────────────────────────────────────────────────

  public void Dispose()
  {
    _timelineSubscription?.Dispose();
    DeregisterCometSimListener();
    _positionSubject.Dispose();
  }
}
