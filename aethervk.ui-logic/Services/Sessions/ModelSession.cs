namespace AetherVk.Logic.Services;

/// <summary>
/// Holds the form data for the Model tab.
/// Exclusive because the engine scene manages a single active 3-D model configuration.
/// </summary>
[ExclusiveSession]
public sealed class ModelSession : ITabSession
{
  // Placeholder — future fields might include:
  //   public ulong ModelEntityId { get; set; }
  //   public string ModelPath { get; set; } = string.Empty;
}
