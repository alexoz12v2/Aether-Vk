namespace AetherVk.Logic.Messages;

public class TransformUpdatedFromNativeMessage
{
  public ulong SceneId { get; }
  public ulong EntityId { get; }

  public TransformUpdatedFromNativeMessage(ulong sceneId, ulong entityId)
  {
    SceneId = sceneId;
    EntityId = entityId;
  }
}
