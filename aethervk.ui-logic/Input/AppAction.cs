namespace AetherVk.Logic.Input;

public readonly struct AppAction(string id, object? payload = null)
{
  public string Id { get; } = id;
  // eg System.Numerics.Vector2 for mouse drag, float for scroll, ...
  public object? Payload { get; } = payload;
}

