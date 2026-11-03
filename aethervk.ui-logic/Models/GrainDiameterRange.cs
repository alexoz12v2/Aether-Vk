using System;

namespace AetherVk.Logic.Models;

/// <summary>Why a smallest / largest grain diameter pair cannot be simulated.</summary>
public enum GrainRangeError
{
  None = 0,
  /// <summary>A bound is not a finite number inside <see cref="GrainDiameterRange.MinBoundUm"/> ..
  /// <see cref="GrainDiameterRange.MaxBoundUm"/>.</summary>
  OutOfBounds,
  /// <summary>The smallest diameter is above the largest (never swapped silently).</summary>
  MinAboveMax,
  /// <summary>The largest / smallest ratio is above <see cref="GrainDiameterRange.MaxRatio"/>.</summary>
  RatioTooLarge,
}

/// <summary>
/// The grain size law as the user states it (smallest and largest grain diameter, µm) and as the
/// engine takes it (reference diameter D and range factor F, <c>ParticleSystemEmitParams</c>):
/// D = √(D_min·D_max), F = √(D_max/D_min); back D_min = D/F, D_max = D·F. The same physics, only
/// another way to name the two numbers. D_min = D_max is one size (F = 1).
/// </summary>
public static class GrainDiameterRange
{
  /// <summary>Smallest accepted diameter (µm).</summary>
  public const double MinBoundUm = 0.01;
  /// <summary>Largest accepted diameter (µm).</summary>
  public const double MaxBoundUm = 10000;
  /// <summary>Largest D_max / D_min: the engine clamps F to 1..1000, and F = √ratio.</summary>
  public const double MaxRatio = 1e6;

  public static GrainRangeError Validate(double minUm, double maxUm)
  {
    if (!IsFinite(minUm) || !IsFinite(maxUm) || minUm < MinBoundUm * 0.999999 || maxUm > MaxBoundUm * 1.000001 || maxUm < MinBoundUm * 0.999999
        || minUm > MaxBoundUm * 1.000001)
      return GrainRangeError.OutOfBounds;
    if (minUm > maxUm)
      return GrainRangeError.MinAboveMax;
    if (maxUm / minUm > MaxRatio * (1 + 1e-9))
      return GrainRangeError.RatioTooLarge;
    return GrainRangeError.None;
  }

  /// <summary>The engine's (D, F) of a valid range.</summary>
  public static (float DiametreUm, float SizeRange) ToEngine(double minUm, double maxUm)
  {
    double d = Math.Sqrt(minUm * maxUm);
    double f = Math.Sqrt(maxUm / minUm);
    return ((float)d, (float)Math.Min(Math.Max(f, 1.0), 1000.0));
  }

  /// <summary>The smallest grain diameter of an engine (D, F) pair (6 significant digits).</summary>
  public static float MinOf(double diametreUm, double sizeRange) => Round(diametreUm / Math.Max(sizeRange, 1.0));

  /// <summary>The largest grain diameter of an engine (D, F) pair (6 significant digits).</summary>
  public static float MaxOf(double diametreUm, double sizeRange) => Round(diametreUm * Math.Max(sizeRange, 1.0));

  /// <summary>
  /// The β range of the simulated grains: β ∝ 1/s, and β is the reference grain's (diameter D), so
  /// the largest grain (D·F) has β/F and the smallest (D/F) has β·F.
  /// </summary>
  public static (double Min, double Max) BetaRange(double beta, double sizeRange)
  {
    double f = IsFinite(sizeRange) ? Math.Max(sizeRange, 1.0) : 1.0;
    return (beta / f, beta * f);
  }

  /// <summary>"β_min – β_max" with 3 significant digits (invariant culture); empty while β is not
  /// known (0, before the engine computed it) or not finite.</summary>
  public static string BetaRangeText(double beta, double sizeRange)
  {
    if (!IsFinite(beta) || beta <= 0)
      return "";
    var (lo, hi) = BetaRange(beta, sizeRange);
    var ci = System.Globalization.CultureInfo.InvariantCulture;
    return lo.ToString("G3", ci) + " – " + hi.ToString("G3", ci);
  }

  private static bool IsFinite(double v) => !double.IsNaN(v) && !double.IsInfinity(v);

  // 30 -> D = 54.7723, F = 1.82574 -> 30.0000x in float: shown as the 30 that was typed
  private static float Round(double v) =>
    IsFinite(v) && v != 0 ? (float)double.Parse(v.ToString("G6", System.Globalization.CultureInfo.InvariantCulture),
      System.Globalization.CultureInfo.InvariantCulture) : (float)v;
}

/// <summary>
/// The smallest / largest diameter boxes of one owner (the model defaults, one jet). An invalid pair
/// (smallest above largest, a ratio above 1e6, a bound out of range) is held here, not applied and
/// not swapped: the engine values (D, F) stay at the last valid pair and <see cref="Error"/> says why.
/// </summary>
public sealed class GrainRangeEdit
{
  private float? _min;
  private float? _max;

  public GrainRangeError Error { get; private set; }

  public float Min(float diametreUm, float sizeRange) => _min ?? GrainDiameterRange.MinOf(diametreUm, sizeRange);

  public float Max(float diametreUm, float sizeRange) => _max ?? GrainDiameterRange.MaxOf(diametreUm, sizeRange);

  /// <summary>One box was edited; the engine pair when the pair is now valid, else null.</summary>
  public (float DiametreUm, float SizeRange)? Edit(bool isMax, float value, float diametreUm, float sizeRange)
  {
    float min = Min(diametreUm, sizeRange), max = Max(diametreUm, sizeRange);
    if (isMax) _max = max = value;
    else _min = min = value;
    Error = GrainDiameterRange.Validate(min, max);
    if (Error != GrainRangeError.None) return null;
    _min = _max = null;
    return GrainDiameterRange.ToEngine(min, max);
  }

  /// <summary>The engine pair changed from outside (a save, Copy to all jets): drop the pending edit.</summary>
  public void Reset()
  {
    _min = _max = null;
    Error = GrainRangeError.None;
  }
}
