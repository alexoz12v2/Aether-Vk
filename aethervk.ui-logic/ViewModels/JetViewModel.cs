using System;
using System.Numerics;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// Represents a single dust emission jet, wrapping all editable per-jet properties.
/// All mutable properties raise <see cref="System.ComponentModel.INotifyPropertyChanged"/>
/// so that <see cref="ModelTabViewModel"/> can react and submit a <c>ModifyParticleSystem</c>
/// call after a short debounce.
/// </summary>
public sealed partial class JetViewModel : ObservableObject
{
  // ── Shared RNG (netstandard2.0 has no Random.Shared) ─────────────────────────
  private static readonly Random _rng = new Random();

  // ── Identity ─────────────────────────────────────────────────────────────────

  /// <summary>
  /// 1-based display index set by <see cref="ModelTabViewModel"/> after the jet is added.
  /// </summary>
  public int DisplayIndex { get; internal set; }

  /// <summary>
  /// Entity ID returned by <c>avkSimulationContext_addParticleSystem</c>.
  /// Zero until the native call succeeds.
  /// </summary>
  public ulong NativePsId { get; internal set; }

  /// <summary>
  /// PCG seed — generated once at construction. Shown as an advanced read-only field.
  /// </summary>
  public uint Seed { get; }

  // ── Jet-specific properties ──────────────────────────────────────────────────

  /// <summary>
  /// Latitude of the jet origin on the comet surface, in degrees.
  /// Range: −90° to +90° (but typically −60° to +60° for active regions).
  /// Corresponds to <c>ParticleSystemEmitParams.latitude_rad</c> (converted at call site).
  /// </summary>
  [ObservableProperty]
  private float _latitudeDeg;

  /// <summary>
  /// Longitude of the jet origin on the comet surface, in degrees.
  /// Range: −180° to +180°.
  /// Corresponds to <c>ParticleSystemEmitParams.longitude_rad</c>.
  /// </summary>
  [ObservableProperty]
  private float _longitudeDeg;

  /// <summary>
  /// Half-angle of the emission cone in degrees.
  /// A narrow jet is ~5–10°; a broad active region is ~40–60°.
  /// Corresponds to <c>ParticleSystemEmitParams.aperture_rad</c>.
  /// </summary>
  [ObservableProperty]
  private float _apertureDeg;

  /// <summary>
  /// Mean initial particle speed in m/s. Typical cometary dust ejects at 0.1–2.0 m/s.
  /// Corresponds to <c>ParticleSystemEmitParams.start_velocity_mean</c>.
  /// </summary>
  [ObservableProperty]
  private float _startVelocityMeanMs;

  /// <summary>
  /// Standard deviation of initial particle speed in m/s.
  /// Typically 10–40% of <see cref="StartVelocityMeanMs"/>.
  /// Corresponds to <c>ParticleSystemEmitParams.start_velocity_std</c>.
  /// </summary>
  [ObservableProperty]
  private float _startVelocityStdMs;

  /// <summary>
  /// RGBA stream color [0,1] packed in a Vector4 (X=R, Y=G, Z=B, W=A).
  /// Corresponds to <c>ParticleSystemDrawParams.stream_color</c>.
  /// </summary>
  [ObservableProperty]
  private Vector4 _streamColor = new Vector4(0.4f, 0.8f, 1.0f, 1.0f); // cyan default

  // ── Computed (native round-trip) ─────────────────────────────────────────────

  /// <summary>
  /// β = ratio of radiation pressure to solar gravity.
  /// Populated from <c>ParticleSystemComputedDTO.Beta</c> after each Add/Modify call.
  /// </summary>
  [ObservableProperty]
  private float _beta;

  /// <summary>
  /// Dust production rate in kg/s at 1 AU, derived from Afρ parameters.
  /// Populated from <c>ParticleSystemComputedDTO.DustProductionRateAt1AuKgs</c>.
  /// </summary>
  [ObservableProperty]
  private float _dustProductionRateAt1AuKgs;

  // ── Unit helpers for FFI ─────────────────────────────────────────────────────

  private const float DegToRad = (float)(Math.PI / 180.0);

  internal float LatitudeRad  => LatitudeDeg  * DegToRad;
  internal float LongitudeRad => LongitudeDeg * DegToRad;
  internal float ApertureRad  => ApertureDeg  * DegToRad;

  // ── Construction ─────────────────────────────────────────────────────────────

  /// <summary>
  /// Creates a new <see cref="JetViewModel"/> with physically reasonable random defaults.
  /// Values are different every time so multiple jets are visually distinct.
  /// </summary>
  public JetViewModel()
  {
    // Latitude: active regions between −60° and +60°
    _latitudeDeg = (float)(_rng.NextDouble() * 120.0 - 60.0);

    // Longitude: anywhere on the comet
    _longitudeDeg = (float)(_rng.NextDouble() * 360.0 - 180.0);

    // Aperture: moderately focused cone, 10–40°
    _apertureDeg = (float)(_rng.NextDouble() * 30.0 + 10.0);

    // Velocity mean: 1.5–3.5 m/s, realistic cometary dust for 67P
    _startVelocityMeanMs = (float)(_rng.NextDouble() * 2.0 + 1.5);

    // Velocity std: 15–35% of mean
    _startVelocityStdMs = _startVelocityMeanMs * (float)(_rng.NextDouble() * 0.20 + 0.15);

    // Random hue, fixed saturation/lightness for visibility in the 3D viewport
    float hue = (float)(_rng.NextDouble() * 360.0);
    _streamColor = HslToRgba(hue, 0.75f, 0.70f);

    // PCG seed — random uint, user can note and reuse it for reproducibility
    var buf = new byte[4];
    _rng.NextBytes(buf);
    Seed = BitConverter.ToUInt32(buf, 0);
  }

  // ── Private helpers ───────────────────────────────────────────────────────────

  /// <summary>Converts HSL (h in [0,360], s,l in [0,1]) to RGBA Vector4 with alpha=1.</summary>
  private static Vector4 HslToRgba(float h, float s, float l)
  {
    float c = (1f - (float)Math.Abs(2f * l - 1f)) * s;
    float hPrime = h / 60f;
    float x = c * (1f - (float)Math.Abs(hPrime % 2f - 1f));
    float m = l - c / 2f;

    float r, g, b;
    if      (hPrime < 1f) { r = c; g = x; b = 0; }
    else if (hPrime < 2f) { r = x; g = c; b = 0; }
    else if (hPrime < 3f) { r = 0; g = c; b = x; }
    else if (hPrime < 4f) { r = 0; g = x; b = c; }
    else if (hPrime < 5f) { r = x; g = 0; b = c; }
    else                  { r = c; g = 0; b = x; }

    return new Vector4(r + m, g + m, b + m, 1f);
  }
}
