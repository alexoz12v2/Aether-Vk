namespace AetherVk.Logic.Services;

/// <summary>
/// Holds the application settings managed by the Settings tab.
/// Exclusive because application settings are global.
/// </summary>
[ExclusiveSession]
public sealed class SettingsSession : ITabSession
{
  // ── Earth Observer Mode ──────────────────────────────────────────────────────
  /// <summary>Observer's geodetic latitude in degrees (−90 … +90).</summary>
  public float EarthObserverLatDeg { get; set; } = 0f;

  /// <summary>Observer's longitude in degrees (−180 … +180, positive = East).</summary>
  public float EarthObserverLonDeg { get; set; } = 0f;

  /// <summary>How the camera's look direction behaves in Earth Observer mode.</summary>
  public EarthObserverOrientationMode EarthObserverOrientation { get; set; } =
    EarthObserverOrientationMode.Inertial;
}
