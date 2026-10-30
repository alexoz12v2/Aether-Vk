namespace AetherVk.Logic.Services;

/// <summary>
/// Holds the simulation timeline state (epoch range, playhead position, playback speed).
/// Exclusive because the engine drives a single simulation clock.
/// </summary>
[ExclusiveSession]
public sealed class TimelineSession : ITabSession
{
  public string CommittedStartEpoch { get; set; } = string.Empty;
  public string CommittedEndEpoch { get; set; } = string.Empty;
  
  // the default range lives here, not in the tab: the untouched session is the save-file "dirty"
  // baseline, so a tab seeding it later would mark a fresh application as changed
  public string ProposedStartEpoch { get; set; } = DefaultProposedStart;
  public string ProposedEndEpoch { get; set; } = DefaultProposedEnd;

  // TODO: read defaults from a configuration file in the future.
  public const string DefaultProposedStart = "2025-10-01T00:00:00Z";
  public const string DefaultProposedEnd = "2025-11-11T00:00:00Z";

  /// <summary>
  /// The scale the four texts above are in: "utc" or "tdb" (<c>TimeScaleService</c>); empty: the
  /// chosen scale, not pinned yet (a fresh session). Pinned when a text is typed, committed or
  /// rewritten for a scale switch; a save without it is read as "tdb" (older saves typed TDB).
  /// </summary>
  public string ProposedScale { get; set; } = string.Empty;
  
  public string CurrentEpochString { get; set; } = string.Empty;

  /// <summary>The selected playback speed (<c>TimelineTabViewModel.SimulationSpeed</c> as int),
  /// null before one was chosen.</summary>
  public int? Speed { get; set; }

  /// <summary>Days the jets ignite before the committed start (<c>TimelineService.DustLeadDays</c>);
  /// null: the default (one TTL).</summary>
  public double? DustLeadDays { get; set; }

  /// <summary>True while Run to End seeks to the committed end (the viewport overlay shows a
  /// spinner).</summary>
  public bool IsRunningToEnd { get; set; }

  // ── Snapshot Restore fields ──────────────────────────────────────────────

  /// <summary>ISO string of the epoch when simulation was last started.</summary>
  public string SnapshotStartEpoch  { get; set; } = string.Empty;
  /// <summary>ISO string of the end epoch at simulation start.</summary>
  public string SnapshotEndEpoch    { get; set; } = string.Empty;

  /// <summary>
  /// Raised (on the UI thread) after the whole session was replaced from outside (savefile open /
  /// New): a live tab re-reads every value.
  /// </summary>
  public event System.Action? Restored;

  public void NotifyRestored() => Restored?.Invoke();
}
