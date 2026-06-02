namespace AetherVk.Logic.Messages;

/// <summary>
/// Sent when the comet entity is destroyed from the scene.
/// </summary>
public class CometDestroyedMessage
{
  public ulong SceneId { get; init; }
}
