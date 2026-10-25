using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;

namespace AetherVk.Logic.SaveFile;

/// <summary>What <see cref="SaveFileService.OpenAsync"/> did, for the caller and tests.</summary>
public sealed record OpenResult(
  bool Success,
  SaveIndexError Error = SaveIndexError.None,
  bool StateRestored = false,
  bool Sought = false,
  int ExpiredStates = 0,
  int ExpiredSpk = 0,
  IReadOnlyList<string>? MissingFiles = null);

/// <summary>
/// File › New / Open / Save / Save As (docs/savefile.md). The index is authoritative and is
/// written last and atomically; <c>states/*/scene.bin</c> is a cache used only when its dump
/// version and compatibility hash match. All methods run on the UI thread, one at a time.
/// </summary>
public sealed class SaveFileService
{
  /// <summary>Unused non-latest states and non-committed SPKs older than this are deleted at open.</summary>
  public static readonly TimeSpan Expiry = TimeSpan.FromHours(10);

  private readonly ISaveFileHost _host;
  private readonly ISystemClock _clock;
  private readonly PreferencesService? _preferences;
  private readonly string _appVersion;
  private readonly SemaphoreSlim _busy = new(1, 1);

  private SaveIndex? _index;
  private string? _fingerprint;
  // live asset key → the saved entry it came from (keeps original_source and imported_at)
  private readonly Dictionary<string, AssetDto> _assetOrigins = new(StringComparer.Ordinal);
  private string? _lastReusedHash;

  public SaveFileService(ISaveFileHost host, ISystemClock clock, PreferencesService? preferences = null,
    string appVersion = "")
  {
    _host = host;
    _clock = clock;
    _preferences = preferences;
    _appVersion = appVersion;
    _reuseSubscription = host.ConfigurationChanged.Subscribe(changed => { _ = TryReuseStateAsync(); });
  }

  private readonly IDisposable _reuseSubscription;

  /// <summary>The folder of the open save; null for an unsaved session.</summary>
  public string? CurrentSaveDir { get; private set; }

  /// <summary>The index as last written or opened (null for an unsaved session).</summary>
  public SaveIndex? CurrentIndex => _index;

  /// <summary>Raised after New, Open and Save (the title and the recent list follow).</summary>
  public event Action? Changed;

  /// <summary>Whether a save, open or reuse is in progress.</summary>
  public bool IsBusy => _busy.CurrentCount == 0;

  // ── dirty ──────────────────────────────────────────────────────────────────

  /// <summary>
  /// Whether the session differs from what was last saved or opened: comet, range, model, jets,
  /// assets, appearance, billboards, camera modes and observer sites. The camera pose and the
  /// clock are not compared (they move by themselves while the simulation plays).
  /// </summary>
  /// The first check (at startup) takes the untouched application as the baseline.
  public bool IsDirty
  {
    get
    {
      var now = Fingerprint(_host.Capture());
      _fingerprint ??= now;
      return !string.Equals(now, _fingerprint, StringComparison.Ordinal);
    }
  }

  /// <summary>The comparable part of a snapshot, as JSON.</summary>
  internal static string Fingerprint(SaveSnapshot s)
  {
    var index = new SaveIndex
    {
      FormatVersion = 0,
      Timeline = new TimelineDto
      {
        CommittedStart = s.Timeline.CommittedStart,
        CommittedEnd = s.Timeline.CommittedEnd,
        ProposedStart = s.Timeline.ProposedStart,
        ProposedEnd = s.Timeline.ProposedEnd,
      },
      Spk = s.Comet is { } c ? [c.Spk] : [],
      Model = s.Model,
      Jets = [.. s.Jets],
      // a viewport still in its fresh state is no change: the startup baseline may be taken
      // before the first viewport exists
      Viewports = [.. s.Viewports.Where(v => !IsFresh(v)).Select(v => new ViewportDto
      {
        Mode = v.Mode,
        Observer = v.Observer is { } o
          ? new ObserverDto { LatDeg = o.LatDeg, LonDeg = o.LonDeg, ElevationM = o.ElevationM, Orientation = o.Orientation }
          : null,
      })],
      Assets = [.. s.Assets.Select(a => new AssetDto { Key = a.Key, ContentHash = SaveIndexMapping.HashHex(a.ContentHash) })],
      CometAppearance = s.Appearance,
      Billboards = [.. s.Billboards],
    };
    return SaveIndexIo.ToJson(index);
  }

  /// <summary>A viewport as a new one starts: the default mode, the observer not yet placed.</summary>
  private static bool IsFresh(ViewportDto v)
  {
    var fresh = new ViewportDto();
    return v.Mode == fresh.Mode
      && (v.Observer is null || (v.Observer.LatDeg == 0 && v.Observer.LonDeg == 0 && v.Observer.ElevationM == 0
        && v.Observer.Orientation == new ObserverDto().Orientation));
  }

  // ── new ────────────────────────────────────────────────────────────────────

  /// <summary>File › New: resets the application and forgets the open save.</summary>
  public async Task NewAsync()
  {
    await _busy.WaitAsync();
    try
    {
      await _host.ResetAsync();
      Forget();
      _fingerprint = Fingerprint(_host.Capture());
    }
    finally
    {
      _busy.Release();
    }
    Changed?.Invoke();
  }

  private void Forget()
  {
    CurrentSaveDir = null;
    _index = null;
    _assetOrigins.Clear();
    _lastReusedHash = null;
  }

  // ── save ───────────────────────────────────────────────────────────────────

  /// <summary>
  /// Saves into <paramref name="saveDir"/> (Save: the open save's folder; Save As: a new one).
  /// SPK and assets are copied in, the scene is dumped to <c>states/&lt;hash&gt;/scene.bin</c>, and
  /// the index is written last, atomically.
  /// </summary>
  public async Task<bool> SaveAsync(string saveDir)
  {
    await _busy.WaitAsync();
    bool ok;
    try
    {
      ok = await SaveCoreAsync(Path.GetFullPath(saveDir));
    }
    finally
    {
      _busy.Release();
    }
    if (ok) Changed?.Invoke();
    return ok;
  }

  private async Task<bool> SaveCoreAsync(string dir)
  {
    var now = _clock.UtcNow;
    var snap = _host.Capture();
    var index = StartingIndexFor(dir, now);
    try
    {
      Directory.CreateDirectory(dir);
      SaveSpk(dir, snap, index, now);
      SaveAssets(dir, snap, index, now);
    }
    catch (Exception e) when (e is IOException or UnauthorizedAccessException)
    {
      _host.Notify("Save", $"Could not write into {dir}: {e.Message}", error: true);
      return false;
    }

    await SaveStateAsync(dir, index, now);

    index.AppVersion = _appVersion;
    index.ModifiedAt = now;
    index.Timeline = snap.Timeline;
    index.Model = snap.Model;
    index.Jets = [.. snap.Jets];
    index.Viewports = [.. snap.Viewports];
    index.CometAppearance = snap.Appearance;
    index.Billboards = [.. snap.Billboards];
    try
    {
      SaveIndexIo.Write(dir, index);
    }
    catch (Exception e) when (e is IOException or UnauthorizedAccessException)
    {
      _host.Notify("Save", $"Could not write the index: {e.Message}", error: true);
      return false;
    }

    CurrentSaveDir = dir;
    _index = index;
    foreach (var a in index.Assets) _assetOrigins[a.Key] = a;
    _fingerprint = Fingerprint(snap);
    _preferences?.AddRecentSave(dir);
    return true;
  }

  /// <summary>The open index when saving in place, an index already in the folder, or a new one.</summary>
  private SaveIndex StartingIndexFor(string dir, DateTimeOffset now)
  {
    if (_index is not null && string.Equals(CurrentSaveDir, dir, StringComparison.Ordinal))
      return _index;
    var (existing, _) = SaveIndexIo.Read(dir);
    if (existing is not null)
      return existing;
    return new SaveIndex { CreatedAt = now };
  }

  private static void SaveSpk(string dir, SaveSnapshot snap, SaveIndex index, DateTimeOffset now)
  {
    foreach (var e in index.Spk) e.Committed = false;
    if (snap.Comet is not { } comet) return;

    var rel = "spk/" + Path.GetFileName(comet.SpkSourcePath);
    var dest = Path.Combine(dir, rel);
    Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
    if (!SamePath(comet.SpkSourcePath, dest) && !SameFile(comet.SpkSourcePath, dest))
      File.Copy(comet.SpkSourcePath, dest, overwrite: true);

    var entry = index.Spk.FirstOrDefault(s => s.File == rel);
    var created = entry?.CreatedAt ?? now;
    if (entry is not null) index.Spk.Remove(entry);
    var spk = comet.Spk;
    index.Spk.Add(new SpkDto
    {
      File = rel,
      NaifId = spk.NaifId,
      Designation = spk.Designation,
      FullName = spk.FullName,
      RecordId = spk.RecordId,
      Coverage = spk.Coverage,
      Sbdb = spk.Sbdb,
      ReferenceOrbitMode = spk.ReferenceOrbitMode,
      NucleusRadiusKm = spk.NucleusRadiusKm,
      Rotation = spk.Rotation,
      CreatedAt = created,
      LastUsedAt = now,
      Committed = true,
    });
  }

  private void SaveAssets(string dir, SaveSnapshot snap, SaveIndex index, DateTimeOffset now)
  {
    var previous = index.Assets.ToDictionary(a => a.Key, StringComparer.Ordinal);
    var assets = new List<AssetDto>();
    foreach (var a in snap.Assets)
    {
      var sub = a.Kind == AssetKind.Mesh ? "assets/meshes" : "assets/textures";
      var exported = _host.ExportAsset(a.Id, Path.Combine(dir, sub));
      if (exported is null)
      {
        _host.Notify("Save", $"Could not export asset \"{a.Label}\"; it is left out of the save.", error: true);
        continue;
      }
      if (!previous.TryGetValue(a.Key, out var origin))
        _assetOrigins.TryGetValue(a.Key, out origin);
      assets.Add(new AssetDto
      {
        Key = a.Key,
        Kind = SaveIndexMapping.Snake(a.Kind),
        Label = a.Label,
        OriginalSource = origin?.OriginalSource is { Length: > 0 } src ? src : a.Key,
        StoredFile = sub + "/" + exported.FileName,
        ContentHash = SaveIndexMapping.HashHex(exported.ContentHash),
        ImportedAt = origin?.ImportedAt ?? now,
        ChannelHint = a.ChannelHint is { } ch ? SaveIndexMapping.Snake(ch) : null,
        BundledTextures = a.BundledTextureKeys is { Count: > 0 } b ? b.ToDictionary(kv => kv.Key, kv => kv.Value, StringComparer.Ordinal) : null,
      });
    }
    index.Assets = assets;

    // stored files of assets no longer imported go away
    var keep = new HashSet<string>(assets.Select(a => Path.GetFullPath(Path.Combine(dir, a.StoredFile))),
      StringComparer.Ordinal);
    foreach (var sub in new[] { "assets/meshes", "assets/textures" })
    {
      var folder = Path.Combine(dir, sub);
      if (!Directory.Exists(folder)) continue;
      foreach (var f in Directory.GetFiles(folder))
        if (!keep.Contains(Path.GetFullPath(f)))
          TryDelete(f);
    }
  }

  private async Task SaveStateAsync(string dir, SaveIndex index, DateTimeOffset now)
  {
    var json = _host.GetCompatibilityJson();
    if (json is null) return;
    var hash = SaveIndexIo.CompatHash(json);
    var rel = $"states/{hash}/scene.bin";
    var file = Path.Combine(dir, rel);
    Directory.CreateDirectory(Path.GetDirectoryName(file)!);
    if (!await _host.DumpSceneAsync(file))
    {
      _host.Notify("Save", "The scene state could not be saved; it will be recomputed when the save is opened.",
        error: true);
      return;
    }
    foreach (var s in index.States) s.Latest = false;
    var entry = index.States.FirstOrDefault(s => s.CompatHash == hash);
    if (entry is null)
    {
      entry = new StateDto { CompatHash = hash, CreatedAt = now };
      index.States.Add(entry);
    }
    entry.CompatibilityJson = json;
    entry.File = rel;
    entry.DumpVersion = _host.SceneDumpVersion;
    entry.ModifiedAt = now;
    entry.LastUsedAt = now;
    entry.Latest = true;
    _lastReusedHash = hash;
  }

  // ── open ───────────────────────────────────────────────────────────────────

  /// <summary>
  /// Opens the save in <paramref name="saveDir"/> (the folder, or its index file): expires stale
  /// cache entries, resets the application, then rebuilds it from the index: assets, model and
  /// jets, comet, appearance, scene state (or a seek), cameras and billboards. Paused.
  /// </summary>
  public async Task<OpenResult> OpenAsync(string saveDir)
  {
    if (string.Equals(Path.GetFileName(saveDir), SaveIndex.FileName, StringComparison.Ordinal))
      saveDir = Path.GetDirectoryName(saveDir)!;
    await _busy.WaitAsync();
    OpenResult result;
    try
    {
      result = await OpenCoreAsync(Path.GetFullPath(saveDir));
    }
    finally
    {
      _busy.Release();
    }
    if (result.Success) Changed?.Invoke();
    return result;
  }

  private async Task<OpenResult> OpenCoreAsync(string dir)
  {
    var (index, error) = SaveIndexIo.Read(dir);
    if (index is null)
    {
      _host.Notify("Open", error switch
      {
        SaveIndexError.Missing => $"No {SaveIndex.FileName} in {dir}.",
        SaveIndexError.NewerFormat => "This save was written by a newer AetherVk; update the application to open it.",
        _ => $"{SaveIndex.FileName} in {dir} is damaged.",
      }, error: true);
      return new OpenResult(false, error);
    }

    var now = _clock.UtcNow;
    var (expiredStates, expiredSpk) = Expire(dir, index, now);
    if (expiredStates + expiredSpk > 0)
    {
      try { SaveIndexIo.Write(dir, index); }
      catch (Exception e) when (e is IOException or UnauthorizedAccessException) { /* read-only save: fine */ }
    }

    await _host.ResetAsync();
    Forget();
    var missing = new List<string>();

    // assets: meshes first (their embedded textures come with them), then the textures not yet loaded
    var byKey = await ImportAssetsAsync(dir, index, missing);
    ulong IdOf(string? key) => key is not null && byKey.TryGetValue(key, out var a) ? a.Id : 0;

    _host.ApplyModel(index.Model, index.Jets);

    var committedSpk = index.Spk.FirstOrDefault(s => s.Committed);
    bool committed = false;
    if (committedSpk is not null && index.Timeline.CommittedStart is { } cs && index.Timeline.CommittedEnd is { } ce)
    {
      var spkPath = Path.Combine(dir, committedSpk.File);
      if (!File.Exists(spkPath))
      {
        missing.Add(committedSpk.File);
        _host.Notify("Open", $"The comet kernel {committedSpk.File} is missing; the comet is not committed.", error: true);
      }
      else
      {
        _host.ApplyTimeline(index.Timeline, committed: false);
        committed = await _host.CommitCometAsync(spkPath, committedSpk, SaveIndexMapping.ToRange(cs, ce));
        if (!committed)
          _host.Notify("Open", $"Committing {committedSpk.FullName} failed.", error: true);
      }
    }
    _host.ApplyTimeline(index.Timeline, committed);

    if (index.CometAppearance is { } appearance)
    {
      var native = SaveIndexMapping.FromDto(appearance, IdOf, out bool fellBack);
      if (fellBack)
        _host.Notify("Open", "The comet mesh of this save could not be loaded; the default nucleus is shown.", error: true);
      _host.ApplyAppearance(native);
    }

    bool restored = false, sought = false;
    if (committed)
    {
      restored = await TryRestoreLatestAsync(dir, index, now);
      if (!restored && index.Timeline.CurrentEpoch is { } epoch)
        sought = await _host.SeekAsync(epoch);
    }

    _host.ApplyViewports(index.Viewports);
    _host.ApplyBillboards([.. index.Billboards.Select(b => Remap(b, byKey))]);
    _host.SyncClock();
    _host.RestartDustFeedbackWhenSettled();

    CurrentSaveDir = dir;
    _index = index;
    foreach (var (liveKey, saved) in byKey.Values.Select(v => (v.Key, v.Saved)))
      _assetOrigins[liveKey] = saved;
    _fingerprint = Fingerprint(_host.Capture());
    _preferences?.AddRecentSave(dir);
    return new OpenResult(true, StateRestored: restored, Sought: sought, ExpiredStates: expiredStates,
      ExpiredSpk: expiredSpk, MissingFiles: missing);
  }

  /// <summary>Deletes stale non-latest states and non-committed SPKs with their files.</summary>
  internal static (int States, int Spk) Expire(string dir, SaveIndex index, DateTimeOffset now)
  {
    var staleStates = index.States.Where(s => !s.Latest && now - s.LastUsedAt > Expiry).ToList();
    foreach (var s in staleStates)
    {
      index.States.Remove(s);
      var file = Path.Combine(dir, s.File);
      TryDelete(file);
      var folder = Path.GetDirectoryName(file);
      if (folder is not null && Directory.Exists(folder) && !Directory.EnumerateFileSystemEntries(folder).Any())
        TryDeleteDir(folder);
    }
    var staleSpk = index.Spk.Where(s => !s.Committed && now - s.LastUsedAt > Expiry).ToList();
    foreach (var s in staleSpk)
    {
      index.Spk.Remove(s);
      // another entry may name the same file
      if (!index.Spk.Any(o => o.File == s.File))
        TryDelete(Path.Combine(dir, s.File));
    }
    return (staleStates.Count, staleSpk.Count);
  }

  private sealed record Mapped(ulong Id, string Key, AssetDto Saved);

  /// <summary>Imports the stored assets; returns saved key → live asset (matched by content hash).</summary>
  private async Task<Dictionary<string, Mapped>> ImportAssetsAsync(string dir, SaveIndex index, List<string> missing)
  {
    var map = new Dictionary<string, Mapped>(StringComparer.Ordinal);
    var ordered = index.Assets
      .OrderBy(a => a.Kind == SaveIndexMapping.Snake(AssetKind.Mesh) ? 0 : 1)
      .ToList();
    var loaded = new HashSet<string>(StringComparer.Ordinal);
    foreach (var a in ordered)
    {
      if (loaded.Contains(a.ContentHash))
        continue; // an embedded texture of a mesh already imported
      var path = Path.Combine(dir, a.StoredFile);
      if (!File.Exists(path) || !await _host.ImportAssetAsync(path))
      {
        missing.Add(a.StoredFile);
        _host.Notify("Open", $"The asset \"{a.Label}\" ({a.StoredFile}) could not be loaded.", error: true);
        continue;
      }
      foreach (var live in _host.ListAssets())
        loaded.Add(SaveIndexMapping.HashHex(live.ContentHash));
    }

    var byHash = new Dictionary<string, LiveAsset>(StringComparer.Ordinal);
    foreach (var live in _host.ListAssets())
    {
      var h = SaveIndexMapping.HashHex(live.ContentHash);
      if (!byHash.ContainsKey(h)) byHash[h] = live;
    }
    foreach (var a in index.Assets)
      if (byHash.TryGetValue(a.ContentHash, out var live) && live.Kind == SaveIndexMapping.ParseSnake(a.Kind, AssetKind.Texture))
        map[a.Key] = new Mapped(live.Id, live.Key, a);
    return map;
  }

  private static BillboardDto Remap(BillboardDto b, Dictionary<string, Mapped> byKey) =>
    byKey.TryGetValue(b.TextureKey, out var m)
      ? new BillboardDto
      {
        Id = b.Id, TextureKey = m.Key, CenterX = b.CenterX, CenterY = b.CenterY, WidthPx = b.WidthPx,
        HeightPx = b.HeightPx, Alpha = b.Alpha, RotationDeg = b.RotationDeg, Visible = b.Visible,
      }
      : b; // its texture is gone: the billboard keeps its spec as a placeholder

  /// <summary>Restores the latest state when its version, hash and file all match.</summary>
  private async Task<bool> TryRestoreLatestAsync(string dir, SaveIndex index, DateTimeOffset now)
  {
    var latest = index.States.FirstOrDefault(s => s.Latest);
    if (latest is null) return false;
    await _host.SettleAsync();
    return await TryRestoreAsync(dir, latest, now);
  }

  private async Task<bool> TryRestoreAsync(string dir, StateDto state, DateTimeOffset now)
  {
    if (state.DumpVersion != _host.SceneDumpVersion) return false;
    var file = Path.Combine(dir, state.File);
    if (!File.Exists(file)) return false;
    var json = _host.GetCompatibilityJson();
    if (json is null || SaveIndexIo.CompatHash(json) != state.CompatHash) return false;
    if (!await _host.RestoreSceneAsync(file)) return false;
    state.LastUsedAt = now;
    _lastReusedHash = state.CompatHash;
    return true;
  }

  // ── runtime reuse ──────────────────────────────────────────────────────────

  /// <summary>
  /// After a change that may alter the compatibility JSON (commit, radius, rotation, jets): when
  /// the open save has a matching, version-compatible state, restores it instead of recomputing.
  /// Not while the simulation plays, nor for the state already in use. True when one was restored.
  /// </summary>
  public async Task<bool> TryReuseStateAsync()
  {
    if (_index is null || CurrentSaveDir is null || _host.IsSimulationRunning) return false;
    if (!await _busy.WaitAsync(0)) return false;
    try
    {
      await _host.SettleAsync();
      var json = _host.GetCompatibilityJson();
      if (json is null) return false;
      var hash = SaveIndexIo.CompatHash(json);
      if (hash == _lastReusedHash) return false;
      var state = _index.States.FirstOrDefault(s => s.CompatHash == hash);
      if (state is null) return false;
      if (!await TryRestoreAsync(CurrentSaveDir, state, _clock.UtcNow)) return false;
      _host.SyncClock();
      _host.RestartDustFeedbackWhenSettled();
      return true;
    }
    finally
    {
      _busy.Release();
    }
  }

  // ── helpers ────────────────────────────────────────────────────────────────

  private static bool SamePath(string a, string b) =>
    string.Equals(Path.GetFullPath(a), Path.GetFullPath(b), StringComparison.Ordinal);

  private static bool SameFile(string a, string b)
  {
    if (!File.Exists(a) || !File.Exists(b)) return false;
    var fa = new FileInfo(a);
    var fb = new FileInfo(b);
    if (fa.Length != fb.Length) return false;
    using var sa = fa.OpenRead();
    using var sb = fb.OpenRead();
    var ba = new byte[81920];
    var bb = new byte[81920];
    while (true)
    {
      int na = sa.Read(ba, 0, ba.Length);
      int nb = ReadFully(sb, bb, na);
      if (na != nb) return false;
      if (na == 0) return true;
      for (int i = 0; i < na; i++)
        if (ba[i] != bb[i]) return false;
    }
  }

  private static int ReadFully(Stream s, byte[] buf, int count)
  {
    int total = 0;
    while (total < count)
    {
      int n = s.Read(buf, total, count - total);
      if (n == 0) break;
      total += n;
    }
    return total;
  }

  private static void TryDelete(string file)
  {
    try { if (File.Exists(file)) File.Delete(file); }
    catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
  }

  private static void TryDeleteDir(string dir)
  {
    try { Directory.Delete(dir); }
    catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
  }
}
