using System;
using System.Numerics;
using System.Reactive.Linq;
using System.Reactive.Subjects;

namespace AetherVk.Logic.Services;

/// <summary>
/// Service dedicated to interacting with <see cref="INativeRuntimeService" /> to track the position
/// of the comet entity when properly configured, and also signal when the comet becomes invalidated
/// </summary>
public class CometPositionTrackerService(INativeRuntimeService runtimeService, ISchedulerProvider schedulerProvider) : IDisposable
{
  // BehaviourSubject which caches the latest value. Init to `null` (no comet)
  private readonly BehaviorSubject<Vector3?> _positionSubject = new(null);

  private readonly ISchedulerProvider _schedulerProvider = schedulerProvider;

  // Exposed position or absent as an IObservable
  public IObservable<Vector3?> CometPosition => _positionSubject.ObserveOn(_schedulerProvider.MainThread);

  // TODO hook with runtime service to update comet position and validity

  // unregister from callbacks
  public void Dispose()
  {

  }
}
