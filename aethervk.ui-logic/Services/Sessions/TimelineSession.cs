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
}
