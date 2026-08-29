using System;
using System.Collections.Concurrent;
using System.Collections.Immutable;
using System.Reactive.Subjects;

namespace AetherVk.Logic.Services;

/// <summary>
/// Immutable snapshot of a single active viewport's native runtime identifiers.
/// </summary>
public readonly record struct ViewportEntry(ulong PresentationEngineId, ulong CameraId);

/// <summary>
/// Interface surface for the ViewportRegistry.
/// </summary>
public interface IViewportRegistry
{
  /// <summary>
  /// Observable stream of the full list of active viewports.
  /// Emits on every registration or deregistration, always on the calling thread.
  /// Consumers that need to update UI should <c>ObserveOn(MainThread)</c> themselves.
  /// </summary>
  IObservable<ImmutableArray<ViewportEntry>> ActiveViewports { get; }

  /// <summary>Registers a newly created viewport. No-op if the entry already exists.</summary>
  void Register(ulong presentationEngineId, ulong cameraId);

  /// <summary>Removes a viewport that is being destroyed. No-op if not found.</summary>
  void Unregister(ulong presentationEngineId);
}

/// <summary>
/// Singleton service that tracks all currently alive viewports by their native runtime IDs.
/// <para>
/// <see cref="ViewModels.Viewport3DViewModel"/> is responsible for calling
/// <see cref="Register"/> inside <c>OnViewportCreated</c> and <see cref="Unregister"/>
/// inside its <c>Dispose</c>.
/// </para>
/// <para>
/// Thread-safe: <see cref="Register"/> and <see cref="Unregister"/> may be called from any
/// thread. The pushed <see cref="ImmutableArray{T}"/> snapshot is always consistent because
/// <see cref="ConcurrentDictionary{TKey,TValue}"/> mutations are atomic and the snapshot is
/// built under no additional lock (minor ABA risk is acceptable for debug display purposes).
/// </para>
/// </summary>
public sealed class ViewportRegistry : IViewportRegistry, IDisposable
{
  private readonly ConcurrentDictionary<ulong, ViewportEntry> _entries = new();

  private readonly BehaviorSubject<ImmutableArray<ViewportEntry>> _subject =
    new(ImmutableArray<ViewportEntry>.Empty);

  /// <inheritdoc/>
  public IObservable<ImmutableArray<ViewportEntry>> ActiveViewports => _subject;

  /// <inheritdoc/>
  public void Register(ulong presentationEngineId, ulong cameraId)
  {
    _entries[presentationEngineId] = new ViewportEntry(presentationEngineId, cameraId);
    Push();
  }

  /// <inheritdoc/>
  public void Unregister(ulong presentationEngineId)
  {
    _entries.TryRemove(presentationEngineId, out _);
    Push();
  }

  private void Push() =>
    _subject.OnNext(_entries.Values.ToImmutableArray());

  /// <summary>Completes the observable so subscribers receive <c>OnCompleted</c>.</summary>
  public void Dispose() => _subject.OnCompleted();
}
