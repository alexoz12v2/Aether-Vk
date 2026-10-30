
using System;
using System.Reactive.Linq;
using System.Reactive.Subjects;

namespace AetherVk.Logic.Services;

/// <summary>
/// Immutable snapshot of a validated simulation epoch range (TAI parts).
/// Only populated once the Rust logic thread confirms success via
/// <c>ExternalState::TimeRange</c> callback.
/// </summary>
public sealed record TimeRange(short StartCenturies, ulong StartNs, short EndCenturies, ulong EndNs);

/// <summary>
/// Manages the simulation timeline. Accepts epoch-range change requests, submits
/// them to the native runtime, and exposes the result only after the
/// <c>ExternalState::TimeRange</c> callback confirms success on the Rust side.
///
/// <para>Callers must pre-validate SPK coverage via
/// <see cref="INativeRuntimeService.CheckAlmanacCoverage"/> before calling
/// <see cref="RequestEpochRange"/>. This service does not enforce a deadline:
/// if Rust rejects the range it emits a breadcrumb and the observable remains
/// unchanged (implicit rollback).</para>
/// </summary>
/// <seealso cref="CameraService" />
/// <seealso cref="CometPositionTrackerService" />
/// <seealso cref="ImportedModelsTrackerService" />
public sealed class TimelineService : IDisposable
{
  private readonly INativeRuntimeService _runtimeService;
  private readonly ISchedulerProvider _schedulerProvider;
  private readonly CometConfigService _cometConfigService;
  private readonly BreadcrumbService _breadcrumbService;

  // Null until first successful timeline commit from the runtime
  private readonly BehaviorSubject<TimeRange?> _timeRangeSubject = new(null);
  private readonly BehaviorSubject<TimeRange?> _proposedTimeRangeSubject = new(null);
  private readonly BehaviorSubject<bool> _isSimulationRunningSubject = new(false);
  private readonly Subject<System.Reactive.Unit> _snapshotRestoreSubject = new();
  private readonly Subject<System.Reactive.Unit> _seekedSubject = new();
  private readonly Subject<DateTimeOffset> _modelRangeRequestSubject = new();
  private readonly Subject<System.Reactive.Unit> _recommitRequestSubject = new();
  private readonly IDisposable _listenerToken;


  public TimelineService(
      INativeRuntimeService runtimeService, 
      ISchedulerProvider schedulerProvider,
      CometConfigService cometConfigService,
      BreadcrumbService breadcrumbService)
  {
    _runtimeService = runtimeService;
    _schedulerProvider = schedulerProvider;
    _cometConfigService = cometConfigService;
    _breadcrumbService = breadcrumbService;

    _listenerToken = runtimeService.RegisterExternalStateListener(
      ExternalStateType.TimeRange,
      HandleTimeRangeCallback);
  }

  // ── Observables ────────────────────────────────────────────────────────────

  /// <summary>
  /// The most recently proposed epoch range (not yet committed to the runtime).
  /// </summary>
  public IObservable<TimeRange?> ProposedTimeRange =>
    _proposedTimeRangeSubject.ObserveOn(_schedulerProvider.MainThread);

  /// <summary>
  /// The most recently confirmed epoch range, or <c>null</c> before any timeline
  /// has been committed. Observed on the main-thread scheduler.
  /// </summary>
  public IObservable<TimeRange?> ValidatedTimeRange =>
    _timeRangeSubject.ObserveOn(_schedulerProvider.MainThread);

  /// <summary>
  /// <c>true</c> once a valid, SPK-covered timeline has been committed by the runtime.
  /// Consumed by <see cref="CometPositionTrackerService"/> to trigger the first
  /// comet position fetch. Observed on the main-thread scheduler.
  /// </summary>
  public IObservable<bool> IsTimelineValid =>
    _timeRangeSubject
      .Select(static r => r is not null)
      .DistinctUntilChanged()
      .ObserveOn(_schedulerProvider.MainThread);

  public IObservable<bool> IsSimulationRunning =>
    _isSimulationRunningSubject.ObserveOn(_schedulerProvider.MainThread);

  public bool IsSimulationRunningValue => _isSimulationRunningSubject.Value;

  public IObservable<System.Reactive.Unit> OnSnapshotRestore =>
    _snapshotRestoreSubject.ObserveOn(_schedulerProvider.MainThread);

  // ── Commands ───────────────────────────────────────────────────────────────

  /// <summary>
  /// Submit a new epoch range to the Rust logic thread.
  ///
  /// <para><b>Pre-condition:</b> caller must have verified almanac coverage via
  /// <see cref="INativeRuntimeService.CheckAlmanacCoverage"/>. Rust will still
  /// validate the range is not trivially invalid (start &lt; end), but will NOT
  /// re-check SPK coverage.</para>
  ///
  /// <para>The <see cref="ValidatedTimeRange"/> observable is updated only when
  /// the <c>ExternalState::TimeRange</c> callback fires. If Rust rejects the command
  /// a breadcrumb is emitted; the observable is left unchanged (implicit rollback).</para>
  /// </summary>
  /// <returns><c>true</c> if the command was successfully enqueued.</returns>
  public bool RequestEpochRange(TimeRange range)
  {
    if (_cometConfigService.IsAlmanacCommittedValue)
    {
        _ = _breadcrumbService.ShowMessageAsync("Timeline Lock", "Cannot change timeline while a comet is committed. Unload the comet first.", TimeSpan.FromSeconds(5), 2);
        return false;
    }

    return _runtimeService.SetEpochRange(
      range.StartCenturies, range.StartNs,
      range.EndCenturies, range.EndNs);
  }

  public bool CheckAlmanacCoverage(int spkId, TimeRange range) =>
    _runtimeService.CheckAlmanacCoverage(spkId, range.StartCenturies, range.StartNs, range.EndCenturies, range.EndNs);
    
  public void ProposeEpochRange(TimeRange range)
  {
    _proposedTimeRangeSubject.OnNext(range);
  }

  public bool Play(int speed)
  {
      _runtimeService.SnapshotSceneSync();
      bool ok = _runtimeService.StartSimulation(speed);
      
      // Defensively recover: if it failed to start, it might be due to a stale pending_cross_sync
      // left over from an interrupted run (e.g. RenderDoc capture deadlock).
      if (!ok)
      {
          _runtimeService.ResetSimulationSync();
          _runtimeService.SnapshotSceneSync();
          ok = _runtimeService.StartSimulation(speed);
      }
      
      if (ok) _isSimulationRunningSubject.OnNext(true);
      return ok;
  }

  public bool Pause()
  {
      bool ok = _runtimeService.PauseSimulationSync();
      if (ok) _isSimulationRunningSubject.OnNext(false);
      return ok;
  }

  /// <summary>
  /// A new speed for the running playback. The core takes one only through a pause (it refuses a
  /// start while the last physics sync is pending), so the engine pauses and starts again; the
  /// running state stays on throughout (a "stopped" in between reads as an external stop and
  /// ends the playback in the UI). False when the restart failed (then it reports the stop).
  /// </summary>
  public bool ChangeSpeed(int speed)
  {
      _runtimeService.PauseSimulationSync();
      bool ok = Play(speed);
      if (!ok) _isSimulationRunningSubject.OnNext(false);
      return ok;
  }

  public bool Reset()
  {
      bool ok = _runtimeService.ResetSimulationSync();
      if (ok) _isSimulationRunningSubject.OnNext(false);
      return ok;
  }

  /// <summary>
  /// The one seek path (Run to End, the timeline's epoch box, a FITS observation epoch): blocks
  /// off the UI thread until the engine confirms, then raises <see cref="Seeked"/> — even when the
  /// wait timed out, since the clock moved anyway.
  /// </summary>
  public async System.Threading.Tasks.Task<bool> SeekAsync(short centuries, ulong nanoseconds)
  {
    bool ok = await System.Threading.Tasks.Task.Run(() => _runtimeService.SeekEpochSync(centuries, nanoseconds));
    _seekedSubject.OnNext(System.Reactive.Unit.Default);
    return ok;
  }

  /// <summary>After every <see cref="SeekAsync"/>: the timeline re-reads the engine clock.</summary>
  public IObservable<System.Reactive.Unit> Seeked =>
    _seekedSubject.ObserveOn(_schedulerProvider.MainThread);

  /// <summary>
  /// "Model range" requests (a FITS row's observation epoch): the Comet tab's range section proposes its N
  /// rotations ending at the epoch (feedback #20).
  /// </summary>
  public IObservable<DateTimeOffset> ModelRangeRequests =>
    _modelRangeRequestSubject.ObserveOn(_schedulerProvider.MainThread);

  public void RequestModelRange(DateTimeOffset endUtc) => _modelRangeRequestSubject.OnNext(endUtc);

  /// <summary>
  /// "Recommit with this range" (the Comet tab's range section, while the range is locked): the Comet tab runs its
  /// Download &amp; Commit with the proposed range (it decommits first, so the lock does not apply).
  /// </summary>
  public IObservable<System.Reactive.Unit> RecommitRequests =>
    _recommitRequestSubject.ObserveOn(_schedulerProvider.MainThread);

  public void RequestRecommit() => _recommitRequestSubject.OnNext(System.Reactive.Unit.Default);

  /// <summary>The longest dust lead time: the jet TTL (JET_TTL_US, 30 d) and the SPK margin before the start.</summary>
  public const double MaxDustLeadDays = 30;

  private double _dustLeadDays = MaxDustLeadDays;

  /// <summary>
  /// How long before the committed start the jets ignite (reviewer report 2026-10-29: igniting at the
  /// start left no tail until Run to End rebuilt the last TTL). The default, one TTL, gives a tail
  /// already formed at the start. Applies to jets created from now on (the next commit).
  /// </summary>
  public double DustLeadDays
  {
    get => _dustLeadDays;
    set => _dustLeadDays = double.IsNaN(value) ? MaxDustLeadDays : Math.Min(MaxDustLeadDays, Math.Max(0, value));
  }

  /// <summary>
  /// The ignition jets are created with: the committed start minus <see cref="DustLeadDays"/>, TDB
  /// µs since J2000; <c>long.MaxValue</c> (the scene clock) without a committed comet.
  /// </summary>
  public long IgnitionUs() => IgnitionUs(_cometConfigService.CommittedInputs?.Range, DustLeadDays);

  internal static long IgnitionUs(TimeRange? committed, double leadDays)
  {
    if (committed is null) return long.MaxValue;
    const long UsPerDay = 86_400L * 1_000_000L;
    long startUs = committed.StartCenturies * 36525L * UsPerDay + (long)(committed.StartNs / 1000UL);
    return startUs - (long)Math.Round(leadDays * UsPerDay);
  }

  public bool SnapshotRestore()
  {
      bool ok = _runtimeService.RestoreSnapshotSync();
      if (ok) 
      {
          _isSimulationRunningSubject.OnNext(false);
          _snapshotRestoreSubject.OnNext(System.Reactive.Unit.Default);
      }
      return ok;
  }

  // ── Internal callback handling ─────────────────────────────────────────────

  // Invoked on the native callback thread — must not block, must not throw.
  private unsafe void HandleTimeRangeCallback(nint dataPtr)
  {
    // dataPtr is valid only for the duration of this call — copy immediately.
    var dto = *(CTimeRange*)dataPtr;
    _timeRangeSubject.OnNext(new TimeRange(
      dto.Centuries[0], dto.Nanoseconds[0],
      dto.Centuries[1], dto.Nanoseconds[1]));
  }

  // ── IDisposable ────────────────────────────────────────────────────────────

  public void Dispose()
  {
    _listenerToken.Dispose();
    _timeRangeSubject.Dispose();
    _proposedTimeRangeSubject.Dispose();
    _isSimulationRunningSubject.Dispose();
    _snapshotRestoreSubject.Dispose();
    _seekedSubject.Dispose();
    _modelRangeRequestSubject.Dispose();
    _recommitRequestSubject.Dispose();
  }
}
