using System.Collections.ObjectModel;
using AetherVk.Logic.ViewModels;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.Services;

/// <summary>
/// Holds the form data for the Model tab.
/// Exclusive because the engine scene manages a single active 3-D model configuration.
/// </summary>
[ExclusiveSession]
public sealed partial class ModelSession : ObservableObject, ITabSession
{
  /// <summary>
  /// "show reference-position error" checkbox. Plain property on purpose: an observable one would
  /// trigger the push of the shared model to every jet.
  /// </summary>
  public bool ShowReferencePositionError { get; set; }
  /// <summary>Dust display stretch softening (Model tab "dust visibility"), visual only.</summary>
  public double DustSoftening { get; set; } = ModelTabDefaults.DustSoftening;
  /// <summary>Dust tracers: bright dots on a sparse set of real particles (Model tab), visual only.</summary>
  public bool DustTracers { get; set; } = false;
  /// <summary>Flow pulses (brightness marks on synchrones), a visual aid; off by default.</summary>
  public bool DustFlow { get; set; } = false;
  /// <summary>Time-lapse factor of the dust flow marks (Model tab), 1 = real motion, visual only.</summary>
  public double DustFlowSpeed { get; set; } = 1.0;

  /// <summary>
  /// The Model tab's manual nucleus radius (km; 0 = use the comet's catalogue radius). Plain
  /// property: the tab pushes it itself (debounced); kept here so it survives the tab and saves.
  /// </summary>
  public float ManualNucleusRadiusKm { get; set; } = 2.0f;

  /// <summary>
  /// Raised after the whole session was replaced from outside (savefile open / New): a live Model
  /// tab re-reads every value and subscribes to the restored jets.
  /// </summary>
  public event System.Action? Restored;

  public void NotifyRestored() => Restored?.Invoke();

  // ── Jet list ───────────────────────────────────────────────────────────────────

  /// <summary>
  /// Live list of configured dust emission jets.
  /// The collection is observable so the ViewModel can bind to it directly.
  /// </summary>
  public ObservableCollection<JetViewModel> Jets { get; } = new();

  // ── Shared model properties (common to all jets) ───────────────────────────────
  // See ParticleSystemEmitParams in particles.rs for units and valid ranges.

  /// <summary>
  /// Diameter of a single dust grain in micrometres (μm).
  /// Physically reasonable range: 0.1–1000 μm. Typical: 1–100 μm.
  /// 67P default: 100 μm (significant large grain population).
  /// Corresponds to <c>ParticleSystemEmitParams.diametre_um</c>.
  /// </summary>
  [ObservableProperty]
  private float _diametreUm = 100f;

  /// <summary>
  /// Volume density of a dust grain in g/cm³.
  /// Range: 0.1–7 g/cm³. Typical cometary dust: 0.5–1.0 g/cm³.
  /// 67P default: 0.533 g/cm³.
  /// Corresponds to <c>ParticleSystemEmitParams.density_gcm3</c>.
  /// </summary>
  [ObservableProperty]
  private float _densityGCm3 = 0.533f;

  /// <summary>
  /// Radiation pressure efficiency Q_pr (enters β only, not the brightness).
  /// Valid range: 0.5–2.0. Default: 1.0 (grey body).
  /// Corresponds to <c>ParticleSystemEmitParams.scattering_efficiency</c>.
  /// </summary>
  [ObservableProperty]
  private float _scatteringEfficiency = 1.0f;

  /// <summary>
  /// Geometric albedo p of the dust: the reflected radiance factor at zero phase (A(0) = 4p of
  /// the Afρ literature). Cometary dust 0.03–0.06; valid range 0.01–0.5. Default: 0.04.
  /// Enters the brightness (τ · p · Φ(α) / r_h²) and the Afρ → production conversion.
  /// Corresponds to <c>ParticleSystemEmitParams.geometric_albedo</c>.
  /// </summary>
  [ObservableProperty]
  private float _geometricAlbedo = 0.04f;

  /// <summary>
  /// Exponent q of the differential grain size law n(s) ∝ s^-q. Valid range 2–5; default 3.5.
  /// Enters the physical speck counts (grains per size bin and their sizes) only: the continuum's
  /// size layout stays the q = 3.5 one.
  /// Corresponds to <c>ParticleSystemEmitParams.size_q</c>.
  /// </summary>
  [ObservableProperty]
  private float _sizeQ = 3.5f;

  /// <summary>
  /// Size range factor F: grain radii from r/F to r·F about the configured radius. Valid range
  /// 1–1000; default 10. Enters the size distribution of every cluster.
  /// Corresponds to <c>ParticleSystemEmitParams.size_range</c>.
  /// </summary>
  [ObservableProperty]
  private float _sizeRange = 10f;

  /// <summary>
  /// Afρ photometric parameter at 1 AU heliocentric distance, in cm.
  /// Range: ~8–1,000,000 cm. Typical active comet: 500–50,000 cm.
  /// 67P default: 100 cm (relatively low activity).
  /// Corresponds to <c>ParticleSystemEmitParams.afrho_0_cm</c>.
  /// </summary>
  [ObservableProperty]
  private float _afrho0Cm = 100f;

  /// <summary>
  /// Afρ power-law decay exponent. Range 1.0–4.0; default 2.0 (quadratic).
  /// Corresponds to <c>ParticleSystemEmitParams.afrho_power</c>.
  /// </summary>
  [ObservableProperty]
  private float _afrhoPower = 2.0f;

  /// <summary>
  /// Heliocentric distance cut-off in AU beyond which emission drops to zero.
  /// Must be ≥ 3 AU. Default: 15 AU.
  /// Corresponds to <c>ParticleSystemEmitParams.afrho_cutoff_au</c>.
  /// </summary>
  [ObservableProperty]
  private float _afrhoCutoffAu = 15.0f;

  /// <summary>
  /// Upper clamp on the Afρ value to prevent infinite emission at small distances, in cm.
  /// Example: 100,000 cm (= 1 km), or leave at default 500,000 cm for hyperactive comets.
  /// Corresponds to <c>ParticleSystemEmitParams.afrho_max_value_cm</c>.
  /// </summary>
  [ObservableProperty]
  private float _afrhoMaxValueCm = 100_000f;
}
