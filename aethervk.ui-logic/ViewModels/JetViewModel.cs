using System;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Numerics;
using AetherVk.Logic.Models;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// Represents a single dust emission jet, wrapping all editable per-jet properties: its site,
/// ejection, colour and its own grain and activity parameters.
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
  /// Why the jet is not in the simulation: set when the core refused its particle system on a
  /// commit (the refusal is retried on the next commit or by the Retry button), null while it is
  /// in the simulation or before any commit.
  /// </summary>
  [ObservableProperty]
  private string? _nativeError;

  /// <summary>
  /// The site is never lit over the committed range, so the jet emits no dust (the text names the
  /// range; <see cref="Services.JetNativeSync.RefreshLitWarning"/>). Refreshed after a commit, a
  /// jet add / edit and a save open; null while lit, not in the simulation or without a commit.
  /// </summary>
  [ObservableProperty]
  private string? _litWarning;

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
  /// Mean initial particle speed in m/s, of the reference grain. Large grains leave at 0.1–2 m/s
  /// (Rosetta, close to the nucleus); the small grains that shape the telescopic coma (fans, the
  /// sunward "U" of a fountain) at tens to hundreds of m/s near perihelion.
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

  /// <summary>
  /// Toggles the jet preview mesh visibility in the rendering scene.
  /// Dust particle emission itself is unaffected by this toggle.
  /// </summary>
  [ObservableProperty]
  private bool _isPreviewVisible = true;

  // ── Grain and activity (per jet) ─────────────────────────────────────────────
  // New jets start from the Model tab's defaults (ModelSession); see ParticleSystemEmitParams in
  // particles.rs for units and valid ranges.

  /// <summary>
  /// Reference grain diameter D in µm, √(D_min·D_max); the size law spans diameters D/F..D·F.
  /// Corresponds to <c>ParticleSystemEmitParams.diametre_um</c>.
  /// </summary>
  [ObservableProperty]
  private float _diametreUm = JetGrain.Default.DiametreUm;

  /// <summary>Grain bulk density in g/cm³. Corresponds to <c>ParticleSystemEmitParams.density_gcm3</c>.</summary>
  [ObservableProperty]
  private float _densityGCm3 = JetGrain.Default.DensityGCm3;

  /// <summary>
  /// Radiation-pressure efficiency Q_pr (0.5–2, enters β only).
  /// Corresponds to <c>ParticleSystemEmitParams.scattering_efficiency</c>.
  /// </summary>
  [ObservableProperty]
  private float _scatteringEfficiency = JetGrain.Default.ScatteringEfficiency;

  /// <summary>
  /// Geometric albedo p of the dust (0.01–0.5): brightness only (not the production rate).
  /// Corresponds to <c>ParticleSystemEmitParams.geometric_albedo</c>.
  /// </summary>
  [ObservableProperty]
  private float _geometricAlbedo = JetGrain.Default.GeometricAlbedo;

  /// <summary>
  /// Size law exponent q (2–5). With <see cref="OrbitSizeLaw"/> on the continuous dust follows it
  /// (or <see cref="SizeQTable"/>); off, the continuum keeps q = 3.5 and only the near specks follow it.
  /// Corresponds to <c>ParticleSystemEmitParams.size_q</c>.
  /// </summary>
  [ObservableProperty]
  private float _sizeQ = JetGrain.Default.SizeQ;

  /// <summary>Size range factor F = √(D_max/D_min) (1–1000). Corresponds to <c>ParticleSystemEmitParams.size_range</c>.</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(BetaRangeText))]
  private float _sizeRange = JetGrain.Default.SizeRange;

  // ── Grain size law stated as the smallest / largest diameter (the UI), stored as (D, F) ───────
  private readonly GrainRangeEdit _grainRange = new();
  private bool _applyingGrainRange;

  /// <summary>
  /// Smallest simulated grain diameter in µm (D/F). Setting it keeps the largest and stores the
  /// engine's (D, F); an invalid pair is not applied (see <see cref="GrainRangeError"/>).
  /// </summary>
  public float MinDiameterUm
  {
    get => _grainRange.Min(DiametreUm, SizeRange);
    set => EditGrainRange(false, value);
  }

  /// <summary>Largest simulated grain diameter in µm (D·F); see <see cref="MinDiameterUm"/>.</summary>
  public float MaxDiameterUm
  {
    get => _grainRange.Max(DiametreUm, SizeRange);
    set => EditGrainRange(true, value);
  }

  /// <summary>Why the typed smallest / largest pair is not applied, None when it is (or was).</summary>
  public GrainRangeError GrainRangeError => _grainRange.Error;

  private void EditGrainRange(bool isMax, float value)
  {
    var pair = _grainRange.Edit(isMax, value, DiametreUm, SizeRange);
    _applyingGrainRange = true;
    try
    {
      if (pair is { } p)
      {
        DiametreUm = p.DiametreUm;
        SizeRange = p.SizeRange;
      }
    }
    finally { _applyingGrainRange = false; }
    NotifyGrainRange();
  }

  private void NotifyGrainRange()
  {
    OnPropertyChanged(nameof(MinDiameterUm));
    OnPropertyChanged(nameof(MaxDiameterUm));
    OnPropertyChanged(nameof(GrainRangeError));
  }

  partial void OnDiametreUmChanged(float value)
  {
    if (_applyingGrainRange) return;
    _grainRange.Reset();
    NotifyGrainRange();
  }

  partial void OnSizeRangeChanged(float value)
  {
    if (_applyingGrainRange) return;
    _grainRange.Reset();
    NotifyGrainRange();
  }

  /// <summary>
  /// This jet's dust mass production rate Q_d at 1 AU in kg/s (its activity; the law's exponent,
  /// cutoff and max are model-wide). It no longer depends on the albedo or the grain size.
  /// Corresponds to <c>ParticleSystemEmitParams.dust_rate_1au_kgs</c>.
  /// </summary>
  [ObservableProperty]
  private float _dustRate1AuKgs = JetGrain.Default.DustRate1AuKgs;

  // ── Dynamic model (per jet) ──────────────────────────────────────────────────

  /// <summary>
  /// "Insolation-weighted emission": the emitted mass is Q_d · ∫max(0, cos i) dt over the day and
  /// the emission instants follow cos i, instead of lit / dark gating. Off = the engine's
  /// behaviour before the option (older saves). Corresponds to
  /// <c>ParticleSystemEmitParams.insolation_weighted</c>.
  /// </summary>
  [ObservableProperty]
  private bool _insolationWeighted = JetGrain.Default.InsolationWeighted;

  /// <summary>
  /// "Orbit-dependent size distribution": the continuous dust follows this jet's q, from
  /// <see cref="SizeQTable"/> along the orbit when it has rows. Off = q 3.5 for the continuum as
  /// before (older saves). Corresponds to <c>ParticleSystemEmitParams.orbit_size_law</c>.
  /// </summary>
  [ObservableProperty]
  private bool _orbitSizeLaw = JetGrain.Default.OrbitSizeLaw;

  /// <summary>
  /// Pre-start dust from the ephemeris: before the start epoch the jet state comes from the SPK
  /// whenever it covers that time (the commit downloads it from start − lead), else from the
  /// reference orbit. Off in older saves. Corresponds to <c>ParticleSystemEmitParams.spk_before_start</c>.
  /// </summary>
  [ObservableProperty]
  private bool _spkBeforeStart = JetGrain.Default.SpkBeforeStart;

  /// <summary>
  /// Dust lifetime in days (1 to 365, default 30): the age at which the dust leaves the
  /// simulation; the emission steps follow it (<see cref="EmissionStepsText"/>). Corresponds to
  /// <c>ParticleSystemEmitParams.ttl_days</c>.
  /// </summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(EmissionStepsText))]
  private float _lifetimeDays = JetGrain.Default.LifetimeDays;

  /// <summary>
  /// The emission steps of the three age tiers for <see cref="LifetimeDays"/>: "2.8 h / 19.7 h /
  /// 6.6 d" at 30 days (each tier's age band / 256 windows). Read-only.
  /// </summary>
  public string EmissionStepsText => DustLifetime.EmissionStepsText(LifetimeDays);

  /// <summary>
  /// The rows of the q table (days from perihelion, q), edited in place by the Model tab; up to
  /// <see cref="Models.SizeQTable.MaxRows"/>. Any edit raises <see cref="SizeQTable"/>.
  /// </summary>
  public ObservableCollection<SizeQRowViewModel> SizeQRows { get; } = new();

  private bool _settingSizeQRows;

  /// <summary>
  /// q along the orbit as an immutable value (sorted, clamped): what the native record and the
  /// save carry. Raised on every row edit, add or removal. Empty = the jet's <see cref="SizeQ"/>.
  /// </summary>
  public SizeQTable SizeQTable
  {
    get
    {
      var rows = new SizeQPoint[SizeQRows.Count];
      for (int i = 0; i < rows.Length; i++)
        rows[i] = new SizeQPoint(SizeQRows[i].DaysFromPerihelion, SizeQRows[i].Q);
      return SizeQTable.From(rows);
    }
    set
    {
      _settingSizeQRows = true;
      try
      {
        foreach (var r in SizeQRows)
          r.PropertyChanged -= OnSizeQRowChanged;
        SizeQRows.Clear();
        foreach (var p in value.Rows)
          SizeQRows.Add(NewRow(p.DaysFromPerihelion, p.Q));
      }
      finally { _settingSizeQRows = false; }
      OnPropertyChanged(nameof(SizeQTable));
      OnPropertyChanged(nameof(CanAddSizeQRow));
    }
  }

  /// <summary>Another row can be added (fewer than <see cref="Models.SizeQTable.MaxRows"/>).</summary>
  public bool CanAddSizeQRow => SizeQRows.Count < SizeQTable.MaxRows;

  private SizeQRowViewModel NewRow(float days, float q)
  {
    var row = new SizeQRowViewModel(days, q);
    row.PropertyChanged += OnSizeQRowChanged;
    return row;
  }

  private void OnSizeQRowChanged(object? sender, PropertyChangedEventArgs e)
  {
    if (!_settingSizeQRows)
      OnPropertyChanged(nameof(SizeQTable));
  }

  /// <summary>
  /// Adds a row after the last one (its time + 30 days, the last q; (0 d, the jet's q) for the
  /// first); nothing at the capacity.
  /// </summary>
  [RelayCommand]
  private void AddSizeQRow()
  {
    if (!CanAddSizeQRow) return;
    float days = 0f, q = SizeQ;
    if (SizeQRows.Count > 0)
    {
      var last = SizeQRows[SizeQRows.Count - 1];
      days = last.DaysFromPerihelion + 30f;
      q = last.Q;
    }
    SizeQRows.Add(NewRow(days, q));
    OnPropertyChanged(nameof(SizeQTable));
    OnPropertyChanged(nameof(CanAddSizeQRow));
  }

  /// <summary>Removes <paramref name="row"/>.</summary>
  [RelayCommand]
  private void RemoveSizeQRow(SizeQRowViewModel? row)
  {
    if (row is null || !SizeQRows.Remove(row)) return;
    row.PropertyChanged -= OnSizeQRowChanged;
    OnPropertyChanged(nameof(SizeQTable));
    OnPropertyChanged(nameof(CanAddSizeQRow));
  }

  /// <summary>
  /// The grain and activity parameters as one value (the dynamic model included); setting it
  /// assigns each property (each one raises its own change notification).
  /// </summary>
  public JetGrain Grain
  {
    get => new(DiametreUm, DensityGCm3, ScatteringEfficiency, GeometricAlbedo, SizeQ, SizeRange, DustRate1AuKgs,
      InsolationWeighted, OrbitSizeLaw, SpkBeforeStart, LifetimeDays, SizeQTable);
    set
    {
      DiametreUm = value.DiametreUm;
      DensityGCm3 = value.DensityGCm3;
      ScatteringEfficiency = value.ScatteringEfficiency;
      GeometricAlbedo = value.GeometricAlbedo;
      SizeQ = value.SizeQ;
      SizeRange = value.SizeRange;
      DustRate1AuKgs = value.DustRate1AuKgs;
      InsolationWeighted = value.InsolationWeighted;
      OrbitSizeLaw = value.OrbitSizeLaw;
      SpkBeforeStart = value.SpkBeforeStart;
      LifetimeDays = value.LifetimeDays;
      SizeQTable = value.SizeQTable;
    }
  }

  // ── Computed (native round-trip) ─────────────────────────────────────────────

  /// <summary>
  /// β = ratio of radiation pressure to solar gravity.
  /// Populated from <c>ParticleSystemComputedDTO.Beta</c> after each Add/Modify call.
  /// </summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(BetaRangeText))]
  private float _beta;

  /// <summary>
  /// "β_min – β_max" of the simulated grains (β/F for the largest, β·F for the smallest), 3
  /// significant digits; empty until the engine computed β.
  /// </summary>
  public string BetaRangeText => GrainDiameterRange.BetaRangeText(Beta, SizeRange);

  /// <summary>
  /// Dust production rate in kg/s at 1 AU as the engine applies it (min(Q_d₀, max)).
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

  /// <summary>A new jet with random site and ejection (as above) and the given grain (the model defaults).</summary>
  public JetViewModel(JetGrain grain) : this()
  {
    Grain = grain;
  }

  /// <summary>
  /// Restores a jet exactly as saved (savefile): every property and the PCG <paramref name="seed"/>,
  /// so the dust it emits — and the scene's compatibility JSON — match the saved one.
  /// </summary>
  public JetViewModel(
    float latitudeDeg,
    float longitudeDeg,
    float apertureDeg,
    float startVelocityMeanMs,
    float startVelocityStdMs,
    Vector4 streamColor,
    bool isPreviewVisible,
    uint seed,
    JetGrain? grain = null)
  {
    _latitudeDeg = latitudeDeg;
    _longitudeDeg = longitudeDeg;
    _apertureDeg = apertureDeg;
    _startVelocityMeanMs = startVelocityMeanMs;
    _startVelocityStdMs = startVelocityStdMs;
    _streamColor = streamColor;
    _isPreviewVisible = isPreviewVisible;
    Seed = seed;
    if (grain is not null)
      Grain = grain;
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

/// <summary>One editable row of <see cref="JetViewModel.SizeQRows"/>: q at a time from perihelion.</summary>
public sealed partial class SizeQRowViewModel : ObservableObject
{
  /// <summary>Days from perihelion (negative before it).</summary>
  [ObservableProperty]
  private float _daysFromPerihelion;

  /// <summary>Exponent q (0 to 6).</summary>
  [ObservableProperty]
  private float _q;

  public SizeQRowViewModel(float daysFromPerihelion, float q)
  {
    _daysFromPerihelion = daysFromPerihelion;
    _q = q;
  }
}
