using System;

namespace AetherVk.Logic.SaveFile;

/// <summary>
/// The engine's former Afρ → dust production conversion, kept only to read saves written before
/// the activity input was the production rate itself (Q_d at 1 AU, kg/s). A pre-Q_d save stores
/// Afρ₀ (cm) per jet; on load it becomes the Q_d₀ the old engine derived from it with that jet's
/// grain and speeds, so an old save keeps its activity. Double precision port of the engine's
/// legacy f64 formula.
/// </summary>
internal static class LegacyAfrho
{
  /// <summary>The size law's exponent the old conversion used, whatever the jet's q (continuum q = 3.5).</summary>
  private const double SizeQ = 3.5;

  /// <summary>
  /// Dust production rate (kg/s) the old engine derived from Afρ <paramref name="afrhoCm"/> with a
  /// grain of reference diameter <paramref name="diametreUm"/> (µm), bulk density
  /// <paramref name="densityGcm3"/> (g/cm³), size range factor <paramref name="sizeRange"/> F and
  /// geometric albedo <paramref name="albedo"/> p, ejected at <paramref name="vMeanMs"/> ± <paramref name="vStdMs"/>.
  /// Q = 2 ρ s₀ v (A/100) / (3 p msf (1 + (σ/v)²)): a negative v gives 0, as does p ≤ 0.
  /// </summary>
  public static double DustRateKgs(double afrhoCm, double diametreUm, double densityGcm3,
    double vMeanMs, double vStdMs, double albedo, double sizeRange)
  {
    if (!(albedo > 0))
      return 0.0;
    if (vMeanMs < 0)
      return 0.0;
    double rhoKgM3 = densityGcm3 * 1e3;
    double s0M = diametreUm * 0.5e-6;
    double afrhoM = afrhoCm / 100.0;
    double ratio = vMeanMs > 0 ? vStdMs / vMeanMs : 0.0;
    double msf = MeanSizeFactor(diametreUm, sizeRange);
    return 2.0 * rhoKgM3 * s0M * vMeanMs * afrhoM / (3.0 * albedo * msf * (1.0 + ratio * ratio));
  }

  /// <summary>
  /// The mean size factor of the size law n(s) ∝ s^−q over s₀/F..s₀·F: I(q−1)/(√s_ref · I(q−1.5)) with
  /// I(e) = ∫ s^e ds (the log form at e = −1) and s_ref = √(ab); F outside [1, 1000] or
  /// non-finite falls back to 10, the reference radius is at least 1e-3 µm.
  /// </summary>
  public static double MeanSizeFactor(double diametreUm, double sizeRange)
  {
    double f = sizeRange;
    if (double.IsNaN(f) || double.IsInfinity(f) || f < 1.0 || f > 1000.0)
      f = 10.0;
    double r = Math.Max(diametreUm / 2.0, 1e-3);
    double a = r / f;
    double b = r * f;
    double sref = Math.Sqrt(a * b);
    double I(double e) =>
      Math.Abs(e + 1.0) < 1e-9 ? Math.Log(b / a) : (Math.Pow(b, e + 1.0) - Math.Pow(a, e + 1.0)) / (e + 1.0);
    return I(2.5 - SizeQ) / (Math.Sqrt(sref) * I(2.0 - SizeQ));
  }
}
