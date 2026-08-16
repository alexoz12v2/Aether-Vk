
using System;
using System.Collections.Immutable;
using System.Reactive.Linq;
using System.Reactive.Subjects;

namespace AetherVk.Logic.Services;

// TODO: move to AetherVk.Logic.Models
public sealed class ImportedModel(ulong id, string name)
{
  public readonly ulong Id = id;
  public readonly string Name = name;
}

// TODO: add once TextureImported ExternalState arm lands in Rust (state_id = 4)
// public sealed class ImportedTexture(ulong id, string name) { ... }

/// <summary>
/// Tracks the list of models (and future textures) imported in the current session.
/// Listens to <c>ExternalState::ModelImported</c> (state_id = 2) from the native runtime.
///
/// <para>Exposes the asset catalogue as a reactive <c>ImmutableArray</c> observable
/// so that both logic and UI layers can observe changes. Observed on the main-thread
/// scheduler so ViewModels can bind directly without extra marshaling.</para>
///
/// - part of the "Companion Runtime Service" group
/// </summary>
/// <seealso cref="CameraService" />
/// <seealso cref="CometPositionTrackerService" />
/// <seealso cref="TimelineService" />
public sealed class ImportedModelsTrackerService : IDisposable
{
  private readonly ISchedulerProvider _schedulerProvider;
  private readonly BehaviorSubject<ImmutableArray<ImportedModel>> _modelsSubject = new([]);
  private readonly IDisposable _listenerToken;

  public ImportedModelsTrackerService(INativeRuntimeService runtimeService, ISchedulerProvider schedulerProvider)
  {
    _schedulerProvider = schedulerProvider;

    _listenerToken = runtimeService.RegisterExternalStateListener(
      ExternalStateType.ModelImported,
      HandleModelImportedCallback);
  }

  // ── Observables ────────────────────────────────────────────────────────────

  /// <summary>
  /// Reactive catalogue of imported models. Observed on the main-thread scheduler.
  /// Each emission is the complete, up-to-date list — subscribers do not need to
  /// track changes themselves.
  /// </summary>
  public IObservable<ImmutableArray<ImportedModel>> ImportedModels =>
    _modelsSubject.ObserveOn(_schedulerProvider.MainThread);

  // Future: public IObservable<ImmutableArray<ImportedTexture>> ImportedTextures { get; }

  // ── Mutation helpers ───────────────────────────────────────────────────────

  private void AddModel(ImportedModel item)
  {
    var next = _modelsSubject.Value.Add(item);
    _modelsSubject.OnNext(next);
  }

  private void RemoveModel(ulong id)
  {
    var current = _modelsSubject.Value;
    for (int i = 0; i < current.Length; i++)
    {
      if (current[i].Id == id)
      {
        _modelsSubject.OnNext(current.RemoveAt(i));
        return;
      }
    }
  }

  // ── Internal callback handling ─────────────────────────────────────────────

  // Invoked on the native callback thread — must not block, must not throw.
  private unsafe void HandleModelImportedCallback(nint dataPtr)
  {
    // dataPtr is valid only for the duration of this call — copy immediately.
    var dto = *(CModelImportedDTO*)dataPtr;
    if (dto.WasSuccessful == 0) return; // failed import — breadcrumb already emitted by Rust

    var name = dto.GetPath();
    // Generate a stable ID: runtime will eventually surface the entity ID via a
    // richer DTO. For now use a deterministic hash of the name as placeholder.
    // TODO (Rust): extend CModelImportedDTO to carry the entity/model ID directly.
    ulong id = (uint)name.GetHashCode();
    AddModel(new ImportedModel(id, name));
  }

  // ── IDisposable ────────────────────────────────────────────────────────────

  public void Dispose()
  {
    _listenerToken.Dispose();
    _modelsSubject.Dispose();
  }
}
