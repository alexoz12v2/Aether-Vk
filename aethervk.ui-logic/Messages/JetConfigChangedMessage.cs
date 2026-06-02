namespace AetherVk.Logic.Messages;

/// <summary>
/// Sent when jet emission circle configuration changes (add, remove, or parameter edit).
/// </summary>
public class JetConfigChangedMessage
{
  public ulong SceneId { get; init; }
}
