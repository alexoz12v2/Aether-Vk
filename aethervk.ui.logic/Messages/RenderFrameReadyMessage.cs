namespace AetherVk.Logic.Messages;

public class RenderFrameReadyMessage
{
  public ulong SceneId { get; }
  public ulong PresentationEngineId { get; }
  public ulong RenderGeneration { get; }

  public RenderFrameReadyMessage(ulong sceneId, ulong presentationEngineId, ulong renderGeneration)
  {
    SceneId = sceneId;
    PresentationEngineId = presentationEngineId;
    RenderGeneration = renderGeneration;
  }
}
