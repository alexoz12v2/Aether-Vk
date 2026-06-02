namespace AetherVk.Logic.Messages;

/// <summary>
/// Sent when an almanac SPK/BPC file is loaded or unloaded, indicating
/// that trajectory data may need to be regenerated.
/// </summary>
public class AlmanacUpdatedMessage
{
  public ulong SceneId { get; init; }
  public string FilePath { get; init; } = "";
  public bool WasLoaded { get; init; }
}
