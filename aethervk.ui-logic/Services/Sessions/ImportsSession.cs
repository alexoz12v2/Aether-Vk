namespace AetherVk.Logic.Services;

/// <summary>
/// Holds the state for the Imports tab (SPK files, textures, imported meshes).
/// Exclusive because the import catalog is global to the scene.
/// </summary>
[ExclusiveSession]
public sealed class ImportsSession : ITabSession
{
  // Placeholder — future fields might include:
  //   public System.Collections.Generic.List<string> ImportedSpkPaths { get; set; } = new();
  //   public System.Collections.Generic.List<string> ImportedTexturePaths { get; set; } = new();
}
