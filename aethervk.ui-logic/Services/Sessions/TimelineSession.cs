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
  
  public string ProposedStartEpoch { get; set; } = string.Empty;
  public string ProposedEndEpoch { get; set; } = string.Empty;
  
  public string CurrentEpochString { get; set; } = string.Empty;

  // ── Snapshot Restore fields ──────────────────────────────────────────────

  /// <summary>ISO string of the epoch when simulation was last started.</summary>
  public string SnapshotStartEpoch  { get; set; } = string.Empty;
  /// <summary>ISO string of the end epoch at simulation start.</summary>
  public string SnapshotEndEpoch    { get; set; } = string.Empty;
}
