using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;

namespace AetherVk.Logic.SaveFile;

/// <summary>Wall-clock time, injectable so expiry is testable.</summary>
public interface ISystemClock
{
  DateTimeOffset UtcNow { get; }
}

public sealed class SystemClock : ISystemClock
{
  public DateTimeOffset UtcNow => DateTimeOffset.UtcNow;
}

/// <summary>An imported asset as the save sees it (key, kind, label, content hash).</summary>
public sealed record LiveAsset(
  ulong Id,
  AssetKind Kind,
  string Key,
  string Label,
  ulong ContentHash,
  TextureChannel? ChannelHint,
  IReadOnlyDictionary<string, string?>? BundledTextureKeys = null);

/// <summary>The committed comet: where its SPK is now, and its index entry (without file and stamps).</summary>
public sealed record LiveComet(string SpkSourcePath, SpkDto Spk);

/// <summary>
/// The live application state the index stores, captured at once on the UI thread. Asset keys
/// (appearance, billboards) are the live keys of <see cref="Assets"/>.
/// </summary>
public sealed record SaveSnapshot(
  TimelineDto Timeline,
  LiveComet? Comet,
  ModelDto Model,
  IReadOnlyList<JetDto> Jets,
  IReadOnlyList<ViewportDto> Viewports,
  IReadOnlyList<LiveAsset> Assets,
  CometAppearanceIndexDto? Appearance,
  IReadOnlyList<BillboardDto> Billboards);

/// <summary>
/// Everything <see cref="SaveFileService"/> needs from the running application. The app adapter
/// (<c>AppSaveFileHost</c>) drives the sessions, services and native runtime; tests use a fake.
/// Calls arrive on the UI thread.
/// </summary>
public interface ISaveFileHost
{
  // ── save ───────────────────────────────────────────────────────────────────

  SaveSnapshot Capture();

  /// <summary>The engine's compatibility JSON of the scene; null without a scene.</summary>
  string? GetCompatibilityJson();

  /// <summary>The scene dump format the engine writes and restores.</summary>
  uint SceneDumpVersion { get; }

  /// <summary>Dumps the scene to <paramref name="file"/>; completes when the engine reports.</summary>
  Task<bool> DumpSceneAsync(string file);

  /// <summary>Exports asset <paramref name="assetId"/> into <paramref name="destDir"/>.</summary>
  ExportedAssetFile? ExportAsset(ulong assetId, string destDir);

  // ── open ───────────────────────────────────────────────────────────────────

  /// <summary>File › New: decommit, clear model, jets, billboards and assets, default cameras, paused.</summary>
  Task ResetAsync();

  /// <summary>Imports one stored asset file; false when the import failed.</summary>
  Task<bool> ImportAssetAsync(string path);

  /// <summary>The assets loaded now, with their content hashes.</summary>
  IReadOnlyList<LiveAsset> ListAssets();

  /// <summary>Restores the model session and its jets (jets are created natively on commit).</summary>
  void ApplyModel(ModelDto model, IReadOnlyList<JetDto> jets);

  /// <summary>Sets the timeline's proposed range (and the committed one when <paramref name="committed"/>).</summary>
  void ApplyTimeline(TimelineDto timeline, bool committed);

  /// <summary>Commits the comet from <paramref name="spkPath"/> with the saved inputs over the saved range.</summary>
  Task<bool> CommitCometAsync(string spkPath, SpkDto spk, TimeRange range);

  /// <summary>Applies the comet appearance (asset ids already mapped).</summary>
  bool ApplyAppearance(CometAppearanceDto appearance);

  /// <summary>Waits until configuration changes just made (radius, jets) reached the scene.</summary>
  Task SettleAsync();

  /// <summary>Restores a scene dump; completes when the engine reports.</summary>
  Task<bool> RestoreSceneAsync(string file);

  /// <summary>Moves the (paused) clock to <paramref name="epoch"/>.</summary>
  Task<bool> SeekAsync(EpochParts epoch);

  void ApplyViewports(IReadOnlyList<ViewportDto> viewports);

  void ApplyBillboards(IReadOnlyList<BillboardDto> billboards);

  /// <summary>Shows the engine's clock (where a restore or seek left it) on the timeline: its
  /// progress and the viewport's date label.</summary>
  void SyncClock();

  /// <summary>Once the restored cameras have reached their saved poses, restarts the engine's
  /// adaptive dust quality feedback, so it measures the settled view (does not wait for it).</summary>
  void RestartDustFeedbackWhenSettled();

  /// <summary>
  /// Fires (on the UI thread, debounced) after a change that may alter the compatibility JSON:
  /// commit, radius, jets. The service then looks for a cached state to reuse.
  /// </summary>
  IObservable<System.Reactive.Unit> ConfigurationChanged { get; }

  /// <summary>Whether the simulation is playing (a cached state is not swapped in then).</summary>
  bool IsSimulationRunning { get; }

  /// <summary>Tells the user something (breadcrumb).</summary>
  void Notify(string title, string message, bool error);
}
