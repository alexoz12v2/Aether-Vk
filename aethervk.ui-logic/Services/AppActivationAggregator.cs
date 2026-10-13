using System;
using System.Collections.Generic;
using System.Reactive.Concurrency;
using System.Reactive.Linq;
using System.Reactive.Subjects;

namespace AetherVk.Logic.Services;

/// <summary>
/// "Is any window of the application active?" from the per-window Activated / Deactivated events.
/// Losing activation is debounced (<see cref="Debounce"/>): focus hopping between two of the
/// application's windows (main window → billboard → main window) deactivates one window before
/// activating the other, which must not hide and re-show the unmanaged windows (XWayland
/// override_redirect billboards hide while the application is in the background).
/// </summary>
public sealed class AppActivationAggregator : IDisposable
{
  public static readonly TimeSpan Debounce = TimeSpan.FromMilliseconds(150);

  // windows do not override Equals: reference identity
  private readonly HashSet<object> _active = new();
  private readonly BehaviorSubject<bool> _raw;
  private readonly object _gate = new();

  /// <summary>Distinct "any window active" values; true immediately, false after <see cref="Debounce"/>.</summary>
  public IObservable<bool> IsAppActive { get; }

  public AppActivationAggregator(IScheduler scheduler, bool initiallyActive = true)
  {
    _raw = new BehaviorSubject<bool>(initiallyActive);
    IsAppActive = _raw
      .Select(active => active ? Observable.Return(true) : Observable.Return(false).Delay(Debounce, scheduler))
      .Switch()
      .DistinctUntilChanged()
      .Replay(1)
      .RefCount();
  }

  public void Activated(object window)
  {
    lock (_gate)
    {
      _active.Add(window);
      _raw.OnNext(true);
    }
  }

  public void Deactivated(object window)
  {
    lock (_gate)
    {
      _active.Remove(window);
      _raw.OnNext(_active.Count > 0);
    }
  }

  /// <summary>A window closed: it can no longer be the active one.</summary>
  public void Closed(object window) => Deactivated(window);

  public void Dispose() => _raw.Dispose();
}
