using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Reactive.Concurrency;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using System.Threading;
using System.Threading.Tasks;
using AetherVk.Logic.Models;

namespace AetherVk.Logic.Services;

/// <summary>Outcome of <see cref="AssetLibraryService.ImportAsync"/>.</summary>
/// <param name="Success">Whether the file was imported (or was already present).</param>
/// <param name="MeshId">Mesh asset produced, 0 if none (texture file or failure).</param>
/// <param name="AddedCount">Assets that did not exist before (0 for a re-import).</param>
public sealed record AssetImportResult(bool Success, ulong MeshId, uint AddedCount);

/// <summary>Outcome of <see cref="AssetLibraryService.RemoveAsync"/>.</summary>
/// <param name="Success">Whether the asset was unloaded (refused while the simulation plays).</param>
/// <param name="Ejected">Comets whose appearance changed (ejected to the sphere / channel cleared).</param>
public sealed record AssetRemoveResult(bool Success, uint Ejected);

/// <summary>
/// Companion runtime service mirroring the native asset library (Imports tab).
///
/// <para>Imports are asynchronous on the native side: <see cref="ImportAsync"/> sends the request
/// with a unique id and completes when <see cref="ExternalStateType.AssetImported"/> carrying that
/// id arrives. After every completion the asset list is re-read; thumbnails are fetched once per
/// asset id and cached. Decoded asset data is cached natively under
/// <c>&lt;session dir&gt;/asset_cache</c> as memory-mapped files, so it shares the session lifetime.</para>
///
/// <para>Meshes and textures are always separate entries: images bundled in a glTF/GLB or an
/// OBJ material become texture assets referenced by <see cref="ImportedAsset.BundledTextures"/>.</para>
///
/// - part of the "Companion Runtime Service" group
/// </summary>
public sealed class AssetLibraryService : IDisposable
{
  /// <summary>File extensions accepted by the native importer.</summary>
  public static readonly string[] MeshExtensions = ["obj", "ply", "gltf", "glb"];
  public static readonly string[] TextureExtensions = ["png", "jpg", "jpeg", "ktx2"];
  public static readonly string[] AllExtensions = [.. MeshExtensions, .. TextureExtensions];

  private readonly INativeRuntimeService _runtimeService;
  private readonly ISchedulerProvider _schedulerProvider;
  private readonly string _cacheDir;
  private readonly IDisposable _listenerToken;
  private readonly IDisposable _removedListenerToken;

  private readonly BehaviorSubject<ImmutableArray<ImportedAsset>> _meshes = new([]);
  private readonly BehaviorSubject<ImmutableArray<ImportedAsset>> _textures = new([]);
  private readonly BehaviorSubject<int> _pendingImports = new(0);
  private readonly BehaviorSubject<int> _pendingOperations = new(0);

  private readonly ConcurrentDictionary<ulong, TaskCompletionSource<AssetImportResult>> _pending = new();
  private readonly ConcurrentDictionary<ulong, TaskCompletionSource<AssetRemoveResult>> _pendingRemovals = new();
  private readonly ConcurrentDictionary<ulong, AssetThumbnail> _thumbnails = new();
  private readonly object _refreshLock = new();
  private long _nextRequestId;
  private bool _disposed;

  public AssetLibraryService(
    INativeRuntimeService runtimeService,
    ISchedulerProvider schedulerProvider,
    ILocalStorageService localStorageService)
  {
    _runtimeService = runtimeService;
    _schedulerProvider = schedulerProvider;
    _cacheDir = localStorageService.GetSessionPath("asset_cache");

    _listenerToken = runtimeService.RegisterExternalStateListener(
      ExternalStateType.AssetImported,
      HandleAssetImportedCallback);
    _removedListenerToken = runtimeService.RegisterExternalStateListener(
      ExternalStateType.AssetRemoved,
      HandleAssetRemovedCallback);
  }

  // ── Observables ────────────────────────────────────────────────────────────

  /// <summary>Imported meshes in import order. Observed on the main thread.</summary>
  public IObservable<ImmutableArray<ImportedAsset>> Meshes =>
    _meshes.ObserveOn(_schedulerProvider.MainThread);

  /// <summary>Imported textures in import order. Observed on the main thread.</summary>
  public IObservable<ImmutableArray<ImportedAsset>> Textures =>
    _textures.ObserveOn(_schedulerProvider.MainThread);

  /// <summary>Whether at least one import is in flight. Observed on the main thread.</summary>
  public IObservable<bool> IsImporting =>
    _pendingImports.Select(n => n > 0).DistinctUntilChanged().ObserveOn(_schedulerProvider.MainThread);

  public bool IsImportingValue => _pendingImports.Value > 0;

  /// <summary>
  /// Whether an import or an unload is in flight (the native library is locked meanwhile).
  /// Observed on the main thread.
  /// </summary>
  public IObservable<bool> IsBusy =>
    _pendingOperations.Select(n => n > 0).DistinctUntilChanged().ObserveOn(_schedulerProvider.MainThread);

  public bool IsBusyValue => _pendingOperations.Value > 0;

  public ImmutableArray<ImportedAsset> CurrentMeshes => _meshes.Value;
  public ImmutableArray<ImportedAsset> CurrentTextures => _textures.Value;

  // ── Commands ───────────────────────────────────────────────────────────────

  /// <summary>Whether <paramref name="path"/> has an extension the importer accepts.</summary>
  public static bool IsSupported(string path)
  {
    var ext = Path.GetExtension(path).TrimStart('.').ToLowerInvariant();
    return AllExtensions.Contains(ext);
  }

  /// <summary>
  /// Imports a mesh or texture file. Re-importing a known file (or identical content) creates
  /// no new asset and reports <c>AddedCount == 0</c>.
  /// </summary>
  public Task<AssetImportResult> ImportAsync(string path, CancellationToken cancellationToken = default)
  {
    if (!IsSupported(path))
      return Task.FromResult(new AssetImportResult(false, 0, 0));

    ulong requestId = (ulong)Interlocked.Increment(ref _nextRequestId);
    var tcs = new TaskCompletionSource<AssetImportResult>(TaskCreationOptions.RunContinuationsAsynchronously);
    _pending[requestId] = tcs;
    UpdatePendingCount();

    if (!_runtimeService.ImportAsset(requestId, path, _cacheDir))
    {
      _pending.TryRemove(requestId, out _);
      UpdatePendingCount();
      return Task.FromResult(new AssetImportResult(false, 0, 0));
    }

    if (cancellationToken.CanBeCanceled)
    {
      var reg = cancellationToken.Register(() =>
      {
        if (_pending.TryRemove(requestId, out var t))
        {
          UpdatePendingCount();
          t.TrySetCanceled(cancellationToken);
        }
      });
      tcs.Task.ContinueWith(_ => reg.Dispose(), TaskScheduler.Default);
    }
    return tcs.Task;
  }

  /// <summary>
  /// Unloads an asset. The runtime refuses while the simulation plays; a comet displaying the
  /// asset is ejected to the procedural sphere (mesh) or has the channel cleared (texture).
  /// Textures bundled with an unloaded mesh stay in the library.
  /// </summary>
  public Task<AssetRemoveResult> RemoveAsync(ulong assetId)
  {
    if (assetId == 0)
      return Task.FromResult(new AssetRemoveResult(false, 0));

    ulong requestId = (ulong)Interlocked.Increment(ref _nextRequestId);
    var tcs = new TaskCompletionSource<AssetRemoveResult>(TaskCreationOptions.RunContinuationsAsynchronously);
    _pendingRemovals[requestId] = tcs;
    UpdatePendingCount();

    if (!_runtimeService.RemoveAsset(requestId, assetId))
    {
      _pendingRemovals.TryRemove(requestId, out _);
      UpdatePendingCount();
      return Task.FromResult(new AssetRemoveResult(false, 0));
    }
    return tcs.Task;
  }

  /// <summary>Looks up an asset (mesh or texture) by id.</summary>
  public ImportedAsset? Find(ulong id) =>
    id == 0
      ? null
      : _meshes.Value.FirstOrDefault(a => a.Id == id) ?? _textures.Value.FirstOrDefault(a => a.Id == id);

  /// <summary>
  /// Re-reads the native asset list (thumbnails fetched once per id). Returns <c>false</c> if an
  /// import held the library; the next <see cref="ExternalStateType.AssetImported"/> retries.
  /// </summary>
  public bool Refresh()
  {
    lock (_refreshLock)
    {
      if (_disposed) return false;
      var assets = _runtimeService.GetAssets();
      if (assets is null) return false;

      var withThumbs = new List<ImportedAsset>(assets.Length);
      foreach (var asset in assets)
      {
        if (!_thumbnails.TryGetValue(asset.Id, out var thumb))
        {
          thumb = _runtimeService.GetAssetThumbnail(asset.Id);
          if (thumb is not null) _thumbnails[asset.Id] = thumb;
        }
        withThumbs.Add(asset with { Thumbnail = thumb });
      }

      // forget previews of unloaded assets (ids are never reused)
      foreach (var id in _thumbnails.Keys)
        if (!assets.Any(a => a.Id == id)) _thumbnails.TryRemove(id, out _);

      var meshes = withThumbs.Where(a => a.Kind == AssetKind.Mesh).ToImmutableArray();
      var textures = withThumbs.Where(a => a.Kind == AssetKind.Texture).ToImmutableArray();
      if (!meshes.SequenceEqual(_meshes.Value)) _meshes.OnNext(meshes);
      if (!textures.SequenceEqual(_textures.Value)) _textures.OnNext(textures);
      return true;
    }
  }

  // ── Internal callback handling ─────────────────────────────────────────────

  // Invoked on the native callback thread — must not block, must not throw.
  private unsafe void HandleAssetImportedCallback(nint dataPtr)
  {
    var dto = *(CAssetImportedDTO*)dataPtr; // valid only during this call: copy
    var result = new AssetImportResult(dto.Success != 0, dto.MeshId, dto.AddedCount);

    // The native library lock is released before the callback fires; refresh off the
    // callback thread, then complete the request so awaiters observe the updated list.
    _schedulerProvider.Background.Schedule(() =>
    {
      if (result.Success) Refresh();
      if (_pending.TryRemove(dto.RequestId, out var tcs))
      {
        UpdatePendingCount();
        tcs.TrySetResult(result);
      }
    });
  }

  // Invoked on the native callback thread — must not block, must not throw.
  private unsafe void HandleAssetRemovedCallback(nint dataPtr)
  {
    var dto = *(CAssetRemovedDTO*)dataPtr; // valid only during this call: copy
    var result = new AssetRemoveResult(dto.Success != 0, dto.Ejected);
    _schedulerProvider.Background.Schedule(() =>
    {
      if (result.Success) Refresh();
      if (_pendingRemovals.TryRemove(dto.RequestId, out var tcs))
      {
        UpdatePendingCount();
        tcs.TrySetResult(result);
      }
    });
  }

  private void UpdatePendingCount()
  {
    if (_disposed) return;
    _pendingImports.OnNext(_pending.Count);
    _pendingOperations.OnNext(_pending.Count + _pendingRemovals.Count);
  }

  // ── IDisposable ────────────────────────────────────────────────────────────

  public void Dispose()
  {
    lock (_refreshLock)
    {
      if (_disposed) return;
      _disposed = true;
    }
    _listenerToken.Dispose();
    _removedListenerToken.Dispose();
    foreach (var kv in _pending)
      kv.Value.TrySetCanceled();
    _pending.Clear();
    foreach (var kv in _pendingRemovals)
      kv.Value.TrySetCanceled();
    _pendingRemovals.Clear();
    _meshes.Dispose();
    _textures.Dispose();
    _pendingImports.Dispose();
    _pendingOperations.Dispose();
  }
}
