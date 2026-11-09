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
  // asset id → the cube plane it shows (absent: the import's plane 0) or is being switched to
  private readonly Dictionary<ulong, int> _planes = [];
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
  /// the library is busy: ask again, <see cref="RetryMetadata"/>, or after the next textures
  /// change). Never cached: "not FITS" and "busy" look alike (a re-stretch holds the library too).</summary>
  public FitsMetadata? GetMetadata(ulong assetId)
  {
    var json = _runtime.GetFitsMetadataJson(assetId);
    return json is null ? null : FitsMetadata.Parse(json);
  }

  /// <summary>How many times <see cref="RetryMetadata"/> asks again after a null answer.</summary>
  public const int MetadataRetryAttempts = 5;

  /// <summary>The wait before each of <see cref="RetryMetadata"/>'s attempts.</summary>
  public static readonly TimeSpan MetadataRetryDelay = TimeSpan.FromMilliseconds(200);

  /// <summary>The texture's file name has a FITS extension (<c>.fit</c>, <c>.fits</c>, <c>.fts</c>,
  /// <c>.fz</c>, also <c>.fits.fz</c>): a null metadata answer for it is more likely "busy" than
  /// "not FITS".</summary>
  public static bool LooksLikeFits(ImportedAsset texture) => HasFitsExtension(texture.Key) || HasFitsExtension(texture.Label);

  /// <summary>The path or name ends in a FITS extension (case-insensitive).</summary>
  public static bool HasFitsExtension(string? path)
  {
    if (string.IsNullOrEmpty(path))
      return false;
    var p = path!.Trim();
    foreach (var ext in FitsExtensions)
      if (p.EndsWith(ext, StringComparison.OrdinalIgnoreCase))
        return true;
    return false;
  }

  private static readonly string[] FitsExtensions = [".fits", ".fit", ".fts", ".fz"];

  /// <summary>
  /// Asks for texture <paramref name="assetId"/>'s metadata again on <paramref name="scheduler"/>,
  /// up to <see cref="MetadataRetryAttempts"/> times <see cref="MetadataRetryDelay"/> apart, until
  /// it is found; <paramref name="found"/> then runs once (on the scheduler). The native read
  /// answers "busy" (an import or a re-stretch holds the asset library) like "not FITS", so only a
  /// FITS file name should be retried. Disposing the result stops the attempts.
  /// </summary>
  public IDisposable RetryMetadata(ulong assetId, IScheduler scheduler, Action<FitsMetadata> found, int attempts = MetadataRetryAttempts)
  {
    var serial = new SerialDisposable();
    var done = new BooleanDisposable();
    void Attempt(int left)
    {
      if (done.IsDisposed)
        return;
      FitsMetadata? meta = null;
      try
      {
        meta = GetMetadata(assetId);
      }
      catch (Exception)
      {
        // a failing native call is retried like a busy one
      }
      if (meta is not null)
      {
        found(meta);
        return;
      }
      if (left > 1)
        serial.Disposable = scheduler.Schedule(MetadataRetryDelay, () => Attempt(left - 1));
    }
    if (attempts > 0)
      serial.Disposable = scheduler.Schedule(MetadataRetryDelay, () => Attempt(attempts));
    return new CompositeDisposable(done, serial);
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

  /// <summary>
  /// Shows plane <paramref name="plane"/> (0-based flat index, NAXIS3 fastest) of cube texture
  /// <paramref name="assetId"/> on the background scheduler (the engine re-reads the source file and
  /// keeps the stretch mode), then has the billboard windows copy its texels again, as a re-stretch
  /// does. With <paramref name="textureKey"/>, every billboard of that texture records the plane (saved,
  /// re-applied after a load). False when the engine refused the plane.
  /// </summary>
  public Task<bool> ApplyPlaneAsync(ulong assetId, int plane, string? textureKey = null)
  {
    if (plane < 0)
      return Task.FromResult(false);
    var tcs = new TaskCompletionSource<bool>();
    lock (_gate)
    {
      _planes[assetId] = plane;
      if (textureKey is not null)
        foreach (var b in _billboards.Current.Where(b => b.TextureKey == textureKey && b.FitsPlane != plane))
          _billboards.Update(b.Id, x => x with { FitsPlane = plane });
    }
    _schedulers.Background.Schedule(() =>
    {
      bool ok = false;
      try
      {
        ok = _runtime.SetFitsPlane(assetId, (ulong)plane);
        if (!ok)
          lock (_gate)
            _planes.Remove(assetId); // unknown again: the next reconcile retries
        else
          _assets.NotifyTexelsChanged(assetId);
      }
      finally
      {
        tcs.TrySetResult(ok);
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
      // the plane first (it keeps the stretch mode; a later re-stretch renders on the new plane)
      int wantedPlane = group.Select(b => b.FitsPlane).LastOrDefault(p => p > 0);
      int renderedPlane = _planes.TryGetValue(texture.Id, out int rp) ? rp : 0;
      if (wantedPlane != renderedPlane)
        _ = ApplyPlaneAsync(texture.Id, wantedPlane);
      if (wanted.SameAs(rendered ?? FitsStretch.Default))
        continue;
      _ = ApplyAsync(texture.Id, wanted);
    }
  }

  public void Dispose() => _disposables.Dispose();
}
