namespace AetherVk.Logic.Messages;

public class SimulationStateUpdatedMessage
{
  public ulong SceneId { get; }

  public SimulationStateUpdatedMessage(ulong sceneId)
  {
    SceneId = sceneId;
  }
}
