namespace AetherVk.Logic.Services;

/// <summary>
/// Holds the form data for the Comet tab.
/// Decorated with <see cref="ExclusiveSessionAttribute"/> because the engine scene
/// contains exactly one comet nucleus.
/// </summary>
[ExclusiveSession]
public sealed class CometSession : ITabSession
{
  public int? SpkId { get; set; }
}
