namespace AetherVk.Logic.Models;

/// <summary>
/// The grain and activity parameters one jet owns (see <c>ParticleSystemEmitParams</c> in
/// particles.rs for units and valid ranges): every jet emits its own grains with its own Q_d₀.
/// The Q_d law's shape (exponent, cutoff, max) stays model-wide in <c>ModelSession</c>.
/// </summary>
/// <param name="DiametreUm">Reference grain diameter D (µm), the geometric mean of the smallest and largest grain diameters (D = √(D_min·D_max)); the size law spans diameters D/F..D·F.</param>
/// <param name="DensityGCm3">Grain bulk density (g/cm³).</param>
/// <param name="ScatteringEfficiency">Radiation-pressure efficiency Q_pr (enters β only).</param>
/// <param name="GeometricAlbedo">Geometric albedo p of the dust.</param>
/// <param name="SizeQ">Size law exponent q: the continuous dust follows it with <paramref name="OrbitSizeLaw"/> on (when the table is empty); off, the continuum keeps q = 3.5 and only the near specks follow it.</param>
/// <param name="SizeRange">Size range factor F = √(D_max/D_min); the UI states the law as D_min / D_max (<see cref="GrainDiameterRange"/>).</param>
/// <param name="DustRate1AuKgs">Dust mass production rate Q_d at 1 AU (kg/s): this jet's activity. It no
/// longer depends on the albedo or the grain size (those set the brightness and β only).</param>
/// <param name="InsolationWeighted">"Insolation-weighted emission": the emitted mass follows cos(incidence) over the day instead of lit / dark gating. Off in older saves.</param>
/// <param name="OrbitSizeLaw">"Orbit-dependent size distribution": the continuous dust follows this jet's q (<see cref="SizeQTable"/> along the orbit when it has rows). Off in older saves.</param>
/// <param name="SpkBeforeStart">Pre-start dust from the ephemeris: before the start epoch the jet follows the SPK when it covers the time, else the reference orbit. On for new jets, off in older saves (their SPK covers the lead, so the flag keeps their dust unchanged).</param>
/// <param name="LifetimeDays">Dust lifetime in days (<see cref="DustLifetime.MinDays"/> to <see cref="DustLifetime.MaxDays"/>, default <see cref="DustLifetime.DefaultDays"/>).</param>
/// <param name="SizeQTable">q along the orbit, used with <paramref name="OrbitSizeLaw"/> on; empty = the jet's <paramref name="SizeQ"/>.</param>
public sealed record JetGrain(
  float DiametreUm,
  float DensityGCm3,
  float ScatteringEfficiency,
  float GeometricAlbedo,
  float SizeQ,
  float SizeRange,
  float DustRate1AuKgs,
  bool InsolationWeighted = false,
  bool OrbitSizeLaw = false,
  bool SpkBeforeStart = true,
  float LifetimeDays = DustLifetime.DefaultDays,
  SizeQTable SizeQTable = null!)
{
  /// <summary>The q table, never null (<see cref="Models.SizeQTable.Empty"/> when none was given).</summary>
  public SizeQTable SizeQTable { get; init; } = SizeQTable ?? Models.SizeQTable.Empty;

  /// <summary>Smallest simulated grain diameter (µm), D/F.</summary>
  public float MinDiameterUm => GrainDiameterRange.MinOf(DiametreUm, SizeRange);

  /// <summary>Largest simulated grain diameter (µm), D·F.</summary>
  public float MaxDiameterUm => GrainDiameterRange.MaxOf(DiametreUm, SizeRange);

  /// <summary>67P-like defaults: grains of 10-1000 µm (D 100 µm, F 10), 0.8 g/cm³ (the GIADA dust bulk
  /// density, 785 kg/m³, Fulle et al. 2017; 0.533 is the nucleus'), grey body, p 0.04, q 3.5, Q_d₀ 2 kg/s.</summary>
  public static JetGrain Default { get; } = new(100f, 0.8f, 1.0f, 0.04f, 3.5f, 10f, 2f);
}
