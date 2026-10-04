namespace AetherVk.Logic.Models;

/// <summary>
/// Which osculating orbit the comet reference (red) track is drawn from. Mirrors the Rust
/// <c>ReferenceOrbitMode</c> (<c>u32</c>): chosen once at commit.
/// </summary>
public enum ReferenceOrbitMode : uint
{
  /// <summary>SBDB solution elements, osculating at their own (possibly distant) epoch.</summary>
  Sbdb = 0,

  /// <summary>Re-osculated from the SPK state at the simulation start epoch.</summary>
  OsculatingAtStart = 1,
}
