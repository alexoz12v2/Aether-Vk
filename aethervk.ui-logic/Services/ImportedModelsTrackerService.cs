
using System;
using System.Reactive.Subjects;
using System.Collections.Immutable;
using System.Reactive.Linq;

namespace AetherVk.Logic.Services;

// TODO in Model
public sealed class ImportedModel(ulong id, string name)
{
  public readonly ulong Id = id;
  public readonly string Name = name;
}

/// <summary>
/// Service dedicated to interacting with <see cref="INativeRuntimeService" /> to track the
/// list of imported models
/// </summary>
public class ImportedModelsTrackerService(INativeRuntimeService runtimeService, ISchedulerProvider schedulerProvider) : IDisposable
{
  private readonly BehaviorSubject<ImmutableArray<ImportedModel>> _importedModelsSubject = new([]);

  private readonly ISchedulerProvider _schedulerProvider = schedulerProvider;

  /// <summary>
  /// Immutable getter for the reactive list of models.
  ///
  /// Why are we leaning towards an IObservable of ImmutableArray instead of ObservableCollection?
  /// Because of *Thread Safety*. UI Is not allowed to be updated from a background thread. In
  /// Avalonia only the UI thread can do that, so you'd need to wrap everything in
  /// `Dispatcher.UIThread.Post` callback. If we push stuff into a `BehaviorSubject`, exposed as
  /// an observable of array, then the change will be picked up on the UI thread
  /// <summary>
  public IObservable<ImmutableArray<ImportedModel>> ImportedModels => _importedModelsSubject.ObserveOn(_schedulerProvider.MainThread);

  // TODO hook with runtime service to update model list (external state callback)
  private void addModel(ImportedModel item)
  {
    // get current array, create a new one with an additional element, swap the arrays
    var currentArray = _importedModelsSubject.Value;
    var newArray = currentArray.Add(item);
    _importedModelsSubject.OnNext(newArray);
  }

  private void RemoveModel(ImportedModel item)
  {
    var currentArray = _importedModelsSubject.Value;
    var newArray = currentArray.Remove(item);
    _importedModelsSubject.OnNext(newArray);
  }

  // unregister from callbacks
  public void Dispose()
  {

  }
}
