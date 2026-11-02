using System;
using System.Reactive;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Threading;

namespace AetherVk.Logic.Utils;

public static class DisposableExtensions
{
  /// <summary>
  /// Adds the specific disposable to the provided CompositeDisposable
  /// </summary>
  public static void AddDisposableTo(this IDisposable disposable, CompositeDisposable compositeDisposable)
  {
    if (compositeDisposable == null) throw new ArgumentNullException(nameof(compositeDisposable));
    compositeDisposable.Add(disposable);
  }

  /// <summary>
  /// Emits once when <paramref name="token"/> is cancelled (at once if it already is). Use with
  /// <c>TakeUntil</c> to stop a timer that polls the native runtime before the runtime shuts down.
  /// </summary>
  public static IObservable<Unit> WhenCancelled(this CancellationToken token) =>
    Observable.Create<Unit>(observer =>
    {
      var registration = token.Register(() =>
      {
        observer.OnNext(Unit.Default);
        observer.OnCompleted();
      });
      return Disposable.Create(registration.Dispose);
    });
}
