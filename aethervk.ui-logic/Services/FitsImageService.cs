using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Reactive;
using System.Reactive.Concurrency;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.Utils;

namespace AetherVk.Logic.Services;

/// <summary>
/// FITS texture assets (<c>docs/fits_ffi_contract.md</c>): their header metadata, re-stretching
/// them off the UI thread, and keeping the stretch the billboards ask for. The engine renders the
/// import's zscale and does not persist a re-stretch, so whenever a billboard's
/// <see cref="DesktopBillboard.FitsStretch"/> differs from what its texture last rendered (a savefile
/// load, a re-import under a new asset id) the stretch is applied again.
/// </summary>
public sealed class FitsImageService : IDisposable
{
  private readonly INativeRuntimeService _runtime;
  private readonly AssetLibraryService _assets;
  private readonly DesktopBillboardService _billboards;
  private readonly ISchedulerProvider _schedulers;
  private readonly CompositeDisposable _disposables = [];
  private readonly object _gate = new();
  // asset id → the stretch it renders (absent: the import's default) or is being re-stretched to
  private readonly Dictionary<ulong, FitsStretch> _rendered = [];
  // asset ids known not to be FITS (the probe is a native call)

  public FitsImageService(
    INativeRuntimeService runtime,
    AssetLibraryService assets,
    DesktopBillboardService billboards,
    ISchedulerProvider schedulers)
  {
    _runtime = runtime;
    _assets = assets;
    _billboards = billboards;
    _schedulers = schedulers;
    // the emissions only trigger: Reconcile reads the current state (a queued snapshot can be stale)
    billboards.Billboards.Select(_ => Unit.Default)
      .Merge(assets.Textures.Select(_ => Unit.Default))
      .ObserveOn(schedulers.Background)
      .Subscribe(_ => Reconcile())
      .AddDisposableTo(_disposables);
  }

  /// <summary>Header metadata of texture <paramref name="assetId"/>; null when it is not FITS (or
  /// the library is busy: ask again after the next textures change). Never cached: "not FITS" and
  /// "busy" look alike (a re-stretch holds the library too).</summary>
  public FitsMetadata? GetMetadata(ulong assetId)
  {
    var json = _runtime.GetFitsMetadataJson(assetId);
    return json is null ? null : FitsMetadata.Parse(json);
  }

  /// <summary>
  /// Re-stretches texture <paramref name="assetId"/> on the background scheduler, then has the
  /// billboard windows copy its texels again (<see cref="AssetLibraryService.TexelsChanged"/>).
  /// With <paramref name="textureKey"/>, every billboard of that texture records the stretch (in the
  /// same step, so no automatic re-stretch races it). Null when the engine refused the stretch.
  /// </summary>
  public Task<FitsStretchResult?> ApplyAsync(ulong assetId, FitsStretch stretch, string? textureKey = null)
  {
    var tcs = new TaskCompletionSource<FitsStretchResult?>();
    lock (_gate)
    {
      _rendered[assetId] = stretch;
      if (textureKey is not null)
      {
        var stored = stretch.SameAs(FitsStretch.Default) ? null : stretch;
        foreach (var b in _billboards.Current.Where(b => b.TextureKey == textureKey))
          _billboards.Update(b.Id, x => x with { FitsStretch = stored });
      }
    }
    _schedulers.Background.Schedule(() =>
    {
      FitsStretchResult? result = null;
      try
      {
        result = _runtime.SetFitsStretch(assetId, (uint)stretch.Mode, (uint)stretch.Transfer, stretch.P0, stretch.P1);
        if (result is null)
          lock (_gate)
            _rendered.Remove(assetId); // unknown again: the next reconcile retries
        else
          _assets.NotifyTexelsChanged(assetId);
      }
      finally
      {
        tcs.TrySetResult(result);
      }
    });
    return tcs.Task;
  }

  /// <summary>Applies the stretch the billboards of each loaded FITS texture ask for, once.</summary>
  private void Reconcile()
  {
    lock (_gate)
      Reconcile(_billboards.Current, _assets.CurrentTextures);
  }

  private void Reconcile(ImmutableArray<DesktopBillboard> billboards, ImmutableArray<ImportedAsset> textures)
  {
    foreach (var group in billboards.GroupBy(b => b.TextureKey))
    {
      var texture = textures.FirstOrDefault(t => t.Key == group.Key);
      if (texture is null)
        continue;
      // the stretch belongs to the texture; billboards of one texture agree (the last edit wins)
      var wanted = group.Select(b => b.FitsStretch).LastOrDefault(s => s is not null) ?? FitsStretch.Default;
      var rendered = _rendered.TryGetValue(texture.Id, out var r) ? r : null;
      // (only a FITS texture's billboards carry a non-default stretch: no metadata probe, which a
      // busy library would answer "not FITS")
      if (wanted.SameAs(rendered ?? FitsStretch.Default))
        continue;
      _ = ApplyAsync(texture.Id, wanted);
    }
  }

  public void Dispose() => _disposables.Dispose();
}
