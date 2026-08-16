namespace AetherVk.Logic.Services;

/// <summary>
/// Stable discriminators for the <c>comp_foreign_id</c> argument of
/// <c>SIMULATION_CALLBACK</c>. Must mirror the discriminator values emitted
/// by the Rust logic thread in <c>rlib</c>.
///
/// <para><b>Contract:</b> these values are frozen. Never renumber an existing
/// entry without a matching Rust-side change.</para>
/// </summary>
internal static class ComponentForeignId
{
  /// <summary>
  /// <c>HighResTransformComponent</c> — position (f64 ×3), rotation (quat f32), scale (f32 ×3).
  /// </summary>
  public const ulong HighResTransform = 1;

  /// <summary>
  /// <c>CameraProjection</c> — Perspective or Orthographic projection parameters.
  /// </summary>
  public const ulong CameraProjection = 2;

  /// <summary>
  /// Comet nucleus position — <c>Vec3f64</c> (x, y, z) in simulation units (AU).
  /// </summary>
  public const ulong CometPosition = 3;
}
