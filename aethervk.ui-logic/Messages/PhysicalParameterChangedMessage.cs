namespace AetherVk.Logic.Messages;

/// <summary>
/// Broadcast when a jet emission parameter that affects physics changes.
/// TimelineViewModel receives this to invalidate the current snapshot and
/// notify the user that the simulation should be restarted.
/// Visual-only parameters (colour, render radius) do NOT trigger this.
/// </summary>
public class PhysicalParameterChangedMessage
{
  public ulong SceneId { get; init; }
}
