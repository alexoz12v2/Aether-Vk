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
/// <param name="SizeQ">Size law exponent q; the specks follow it, the continuum keeps q = 3.5.</param>
/// <param name="SizeRange">Size range factor F = √(D_max/D_min); the UI states the law as D_min / D_max (<see cref="GrainDiameterRange"/>).</param>
/// <param name="DustRate1AuKgs">Dust mass production rate Q_d at 1 AU (kg/s): this jet's activity. It no
/// longer depends on the albedo or the grain size (those set the brightness and β only).</param>
public sealed record JetGrain(
  float DiametreUm,
  float DensityGCm3,
  float ScatteringEfficiency,
  float GeometricAlbedo,
  float SizeQ,
  float SizeRange,
  float DustRate1AuKgs)
{
  /// <summary>Smallest simulated grain diameter (µm), D/F.</summary>
  public float MinDiameterUm => GrainDiameterRange.MinOf(DiametreUm, SizeRange);

  /// <summary>Largest simulated grain diameter (µm), D·F.</summary>
  public float MaxDiameterUm => GrainDiameterRange.MaxOf(DiametreUm, SizeRange);

  /// <summary>67P-like defaults: grains of 10-1000 µm (D 100 µm, F 10), 0.8 g/cm³ (the GIADA dust bulk
  /// density, 785 kg/m³, Fulle et al. 2017; 0.533 is the nucleus'), grey body, p 0.04, q 3.5, Q_d₀ 2 kg/s.</summary>
  public static JetGrain Default { get; } = new(100f, 0.8f, 1.0f, 0.04f, 3.5f, 10f, 2f);
}
