using System.Collections.ObjectModel;
using AetherVk.Logic.Models;
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
  /// "show reference-position error" checkbox. Plain property on purpose: the observable ones are
  /// the dust model (the tab re-plots them, and pushes the Q_d law to every jet).
  /// </summary>
  public bool ShowReferencePositionError { get; set; }
  /// <summary>Dust display stretch softening (Model tab "dust visibility"), visual only.</summary>
  public double DustSoftening { get; set; } = ModelTabDefaults.DustSoftening;
  /// <summary>Dust tracers: bright dots on a sparse set of real particles (Model tab), visual only.</summary>
  public bool DustTracers { get; set; } = false;
  /// <summary>Flow pulses (brightness marks on synchrones), a visual aid; off by default.</summary>
  public bool DustFlow { get; set; } = false;
  /// <summary>Trajectory lines: the paths of ~200 grains since they left the nucleus, visual only.</summary>
  public bool DustTrajectories { get; set; } = false;
  /// <summary>Coma 1/ρ normalisation of the dust brightness (display only), off by default.</summary>
  public bool DustRhoNorm { get; set; } = false;
  /// <summary>Dust exposure compensation in stops over the automatic exposure, in [-6, 6], visual only.</summary>
  public double DustExposureEv { get; set; } = 0.0;
  /// <summary>Time-lapse factor of the dust flow marks (Model tab), 1 = real motion, visual only.</summary>
  public double DustFlowSpeed { get; set; } = 1.0;
  /// <summary>The nucleus sphere gizmo is shown (Model tab), visual only.</summary>
  public bool SphereGizmoVisible { get; set; } = true;
  /// <summary>Which sphere gizmo parts are drawn (<see cref="Models.SphereGizmoPart"/> bits, Model tab), visual only.</summary>
  public uint SphereGizmoParts { get; set; } = Models.SphereGizmoPart.Default;
  /// <summary>The selected jet's latitude/longitude arcs and degrees in the viewport (Model tab), visual only.</summary>
  public bool ShowJetAngles { get; set; }

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

  // ── Grain and activity defaults for new jets ────────────────────────────────────
  // Every jet owns its grain and activity parameters (JetViewModel); these are the values a newly
  // added jet starts from, and what a jet of an older save (written before jets had their own)
  // inherits. Editing them changes no existing jet. The Q_d law's shape (DustRatePower,
  // DustRateCutoffAu, DustRateMaxKgs) is model-wide: it is copied into every jet's native parameters.
  // See ParticleSystemEmitParams in particles.rs for units and valid ranges.

  /// <summary>The defaults below as one value (what a new jet starts from).</summary>
  public JetGrain DefaultGrain =>
    new(DiametreUm, DensityGCm3, ScatteringEfficiency, GeometricAlbedo, SizeQ, SizeRange, DustRate1AuKgs);

  /// <summary>
  /// Diameter of a single dust grain in micrometres (μm).
  /// Physically reasonable range: 0.1–1000 μm. Typical: 1–100 μm.
  /// 67P default: 100 μm (significant large grain population).
  /// Corresponds to <c>ParticleSystemEmitParams.diametre_um</c>.
  /// </summary>
  [ObservableProperty]
  private float _diametreUm = JetGrain.Default.DiametreUm;

  /// <summary>
  /// Volume density of a dust grain in g/cm³.
  /// Range: 0.1–7 g/cm³. Typical cometary dust: 0.5–1.0 g/cm³.
  /// 67P default: 0.8 g/cm³ (GIADA dust; 0.533 is the nucleus).
  /// Corresponds to <c>ParticleSystemEmitParams.density_gcm3</c>.
  /// </summary>
  [ObservableProperty]
  private float _densityGCm3 = JetGrain.Default.DensityGCm3;

  /// <summary>
  /// Radiation pressure efficiency Q_pr (enters β only, not the brightness).
  /// Valid range: 0.5–2.0. Default: 1.0 (grey body).
  /// Corresponds to <c>ParticleSystemEmitParams.scattering_efficiency</c>.
  /// </summary>
  [ObservableProperty]
  private float _scatteringEfficiency = JetGrain.Default.ScatteringEfficiency;

  /// <summary>
  /// Geometric albedo p of the dust: the reflected radiance factor at zero phase (A(0) = 4p of
  /// the photometric literature). Cometary dust 0.03–0.06; valid range 0.01–0.5. Default: 0.04.
  /// Enters the brightness (τ · p · Φ(α) / r_h²) only (not the production rate).
  /// Corresponds to <c>ParticleSystemEmitParams.geometric_albedo</c>.
  /// </summary>
  [ObservableProperty]
  private float _geometricAlbedo = JetGrain.Default.GeometricAlbedo;

  /// <summary>
  /// Exponent q of the differential grain size law n(s) ∝ s^-q. Valid range 2–5; default 3.5.
  /// Enters the physical speck counts (grains per size bin and their sizes) only: the continuum's
  /// size layout stays the q = 3.5 one.
  /// Corresponds to <c>ParticleSystemEmitParams.size_q</c>.
  /// </summary>
  [ObservableProperty]
  private float _sizeQ = JetGrain.Default.SizeQ;

  /// <summary>
  /// Size range factor F: grain radii from r/F to r·F about the configured radius. Valid range
  /// 1–1000; default 10. Enters the size distribution of every cluster.
  /// Corresponds to <c>ParticleSystemEmitParams.size_range</c>.
  /// </summary>
  [ObservableProperty]
  private float _sizeRange = JetGrain.Default.SizeRange;

  /// <summary>
  /// Dust mass production rate Q_d at 1 AU heliocentric distance, in kg/s (default for new jets).
  /// It no longer depends on the albedo or the grain size (those set brightness and β only).
  /// Default: 2 kg/s per jet.
  /// Corresponds to <c>ParticleSystemEmitParams.dust_rate_1au_kgs</c>.
  /// </summary>
  [ObservableProperty]
  private float _dustRate1AuKgs = JetGrain.Default.DustRate1AuKgs;

  /// <summary>
  /// Q_d power-law decay exponent k (Q_d ∝ r_h^−k). Range 1.0–4.0; default 2.0. Model-wide (every jet).
  /// Corresponds to <c>ParticleSystemEmitParams.dust_rate_power</c>.
  /// </summary>
  [ObservableProperty]
  private float _dustRatePower = 2.0f;

  /// <summary>
  /// Heliocentric distance cut-off in AU beyond which emission drops to zero.
  /// Must be ≥ 3 AU. Default: 15 AU. Model-wide (every jet).
  /// Corresponds to <c>ParticleSystemEmitParams.dust_rate_cutoff_au</c>.
  /// </summary>
  [ObservableProperty]
  private float _dustRateCutoffAu = 15.0f;

  /// <summary>
  /// Upper clamp on Q_d to prevent infinite emission at small distances, in kg/s.
  /// Default: 2000 kg/s. Model-wide (every jet).
  /// Corresponds to <c>ParticleSystemEmitParams.dust_rate_max_kgs</c>.
  /// </summary>
  [ObservableProperty]
  private float _dustRateMaxKgs = 2000f;
}
