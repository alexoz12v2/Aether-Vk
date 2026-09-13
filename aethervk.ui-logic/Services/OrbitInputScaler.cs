using System;
using System.Numerics;

namespace AetherVk.Logic.Services;

/// <summary>
/// Scales a 2D mouse pixel-delta into an orbit angular delta (radians).
/// Abstracts sensitivity, DPI normalization, and optional quadratic acceleration
/// for the CometOrbiting camera mode.
/// </summary>
public sealed class OrbitInputScaler
{
  /// <summary>Reference dots-per-inch. Adjust to match your display (default 96).</summary>
  public float DotsPerInch { get; set; } = 96f;

  /// <summary>Angular rate in degrees per pixel at reference DPI (default 0.25 °/px).</summary>
  public float SensitivityDegPerPixel { get; set; } = 0.25f;

  /// <summary>
  /// If true, applies a quadratic boost: slow mouse movements stay linear;
  /// fast movements produce proportionally more rotation.
  /// </summary>
  public bool EnableAcceleration { get; set; } = false;

  /// <summary>
  /// Scales a raw pixel delta into an angular delta (radians), normalized for display DPI
  /// and sensitivity setting.
  /// </summary>
  /// <param name="pixelDelta">Raw 2D mouse delta in screen pixels.</param>
  /// <param name="shiftMultiplier">
  /// Additional multiplier applied after DPI/sensitivity scaling.
  /// Typically 0.2 when Shift is held (Blender-style fine control), 1.0 otherwise.
  /// </param>
  /// <returns>Angular delta in radians (X = azimuth change, Y = elevation change).</returns>
  public Vector2 Scale(Vector2 pixelDelta, float shiftMultiplier = 1f)
  {
    const float DegToRad = (float)(Math.PI / 180.0);
    // Higher DPI → more pixels per physical inch → reduce rad/pixel proportionally so the
    // angular rate in rad/inch stays constant across displays.
    float dpiScale = 96f / Math.Max(1f, DotsPerInch);
    float radPerPx = SensitivityDegPerPixel * DegToRad * dpiScale * shiftMultiplier;

    if (!EnableAcceleration)
      return pixelDelta * radPerPx;

    // Quadratic acceleration: v_out = v_in + 0.1 * v_in² (applied as a scalar boost on the direction).
    // At 1 px/frame: 1 + 0.1 = 1.1× (barely noticeable).
    // At 10 px/frame: 10 + 10 = 2× boost → feels snappy for fast sweeps.
    float speed        = pixelDelta.Length();
    float boostedSpeed = speed + 0.1f * speed * speed;
    float accelScale   = speed > 1e-6f ? boostedSpeed / speed : 1f;
    return pixelDelta * (radPerPx * accelScale);
  }
}
