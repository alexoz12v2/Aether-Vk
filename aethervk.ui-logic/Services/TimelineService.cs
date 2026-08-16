
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

  // Null until first successful timeline commit from the runtime
  private readonly BehaviorSubject<TimeRange?> _timeRangeSubject = new(null);
  private readonly IDisposable _listenerToken;

  public TimelineService(INativeRuntimeService runtimeService, ISchedulerProvider schedulerProvider)
  {
    _runtimeService = runtimeService;
    _schedulerProvider = schedulerProvider;

    _listenerToken = runtimeService.RegisterExternalStateListener(
      ExternalStateType.TimeRange,
      HandleTimeRangeCallback);
  }

  // ── Observables ────────────────────────────────────────────────────────────

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
  public bool RequestEpochRange(TimeRange range) =>
    _runtimeService.SetEpochRange(
      range.StartCenturies, range.StartNs,
      range.EndCenturies, range.EndNs);

  public bool CheckAlmanacCoverage(int spkId, TimeRange range) =>
    _runtimeService.CheckAlmanacCoverage(spkId, range.StartCenturies, range.StartNs, range.EndCenturies, range.EndNs);

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
  }
}
