namespace AetherVk.Logic.Models;

/// <summary>
/// Bits of the comet sphere gizmo's drawn parts. Mirrors the Rust <c>SPHERE_GIZMO_PART_*</c>
/// constants (<c>u32</c>, scene.rs).
/// </summary>
public static class SphereGizmoPart
{
  /// <summary>The latitude / longitude wireframe.</summary>
  public const uint Grid = 1u << 0;

  /// <summary>The body-fixed X and Y axes.</summary>
  public const uint BodyAxes = 1u << 1;

  /// <summary>The spin (body Z) axis.</summary>
  public const uint SpinAxis = 1u << 2;

  /// <summary>The arrow towards the Sun.</summary>
  public const uint Sun = 1u << 3;

  /// <summary>The heliocentric velocity direction.</summary>
  public const uint Velocity = 1u << 4;

  /// <summary>What a new comet draws (everything but the velocity).</summary>
  public const uint Default = Grid | BodyAxes | SpinAxis | Sun;

  public const uint All = Default | Velocity;
}
