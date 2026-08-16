using System;
using System.Reactive.Disposables;

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
}
