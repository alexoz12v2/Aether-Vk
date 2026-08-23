using System.Collections.ObjectModel;
using AetherVk.Logic.ViewModels;

namespace AetherVk.Logic.Services;

/// <summary>
/// Holds the form data for the Model tab.
/// Exclusive because the engine scene manages a single active 3-D model configuration.
/// </summary>
[ExclusiveSession]
public sealed class ModelSession : ITabSession
{
  // ── Jet list ─────────────────────────────────────────────────────────────────

  /// <summary>
  /// Live list of configured dust emission jets.
  /// The collection is observable so the ViewModel can bind to it directly.
  /// </summary>
  public ObservableCollection<JetViewModel> Jets { get; } = new();

  // ── Shared model properties (common to all jets) ──────────────────────────────
  // See ParticleSystemEmitParams in particles.rs for units and valid ranges.

  /// <summary>
  /// Percentage [0,1] variability of grain mass.
  /// 0 = all grains have identical mass; 1 = std equals mean.
  /// Corresponds to <c>ParticleSystemEmitParams.mass_variability_perc</c>.
  /// </summary>
  public float MassVariabilityPerc { get; set; } = 0.30f;

  /// <summary>
  /// Diameter of a single dust grain in micrometres (μm).
  /// Physically reasonable range: 0.1–1000 μm. Typical: 1–100 μm.
  /// Corresponds to <c>ParticleSystemEmitParams.diametre_um</c>.
  /// </summary>
  public float DiametreUm { get; set; } = 10f;

  /// <summary>
  /// Volume density of a dust grain in g/cm³.
  /// Range: 0.1–7 g/cm³. Typical cometary dust: 0.5–1.0 g/cm³.
  /// Corresponds to <c>ParticleSystemEmitParams.density_gcm3</c>.
  /// </summary>
  public float DensityGCm3 { get; set; } = 0.5f;

  /// <summary>
  /// Radiation pressure scattering efficiency Q_pr.
  /// Valid range: 0.5–2.0. Default: 1.0 (grey body).
  /// Corresponds to <c>ParticleSystemEmitParams.scattering_efficiency</c>.
  /// </summary>
  public float ScatteringEfficiency { get; set; } = 1.0f;

  /// <summary>
  /// Afρ photometric parameter at 1 AU heliocentric distance, in cm.
  /// Range: ~8–1,000,000 cm. Typical active comet: 500–50,000 cm.
  /// Corresponds to <c>ParticleSystemEmitParams.afrho_0_cm</c>.
  /// </summary>
  public float Afrho0Cm { get; set; } = 500f;

  /// <summary>
  /// Afρ power-law decay exponent. Range 1.0–4.0; default 2.0 (quadratic).
  /// Corresponds to <c>ParticleSystemEmitParams.afrho_power</c>.
  /// </summary>
  public float AfrhoPower { get; set; } = 2.0f;

  /// <summary>
  /// Heliocentric distance cut-off in AU beyond which emission drops to zero.
  /// Must be ≥ 3 AU. Default: 5 AU.
  /// Corresponds to <c>ParticleSystemEmitParams.afrho_cutoff_au</c>.
  /// </summary>
  public float AfrhoCutoffAu { get; set; } = 5.0f;

  /// <summary>
  /// Upper clamp on the Afρ value to prevent infinite emission at small distances, in cm.
  /// Example: 100,000 cm (= 1 km), or leave at default 500,000 cm for hyperactive comets.
  /// Corresponds to <c>ParticleSystemEmitParams.afrho_max_value_cm</c>.
  /// </summary>
  public float AfrhoMaxValueCm { get; set; } = 100_000f;
}
