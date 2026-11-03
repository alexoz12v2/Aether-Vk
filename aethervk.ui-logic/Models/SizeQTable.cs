using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace AetherVk.Logic.Models;

/// <summary>One row of <see cref="SizeQTable"/>: the size law exponent q at a time from perihelion.</summary>
/// <param name="DaysFromPerihelion">Days from perihelion (negative before it).</param>
/// <param name="Q">Exponent q of n(d) ∝ d^−q (<see cref="SizeQTable.MinQ"/> to <see cref="SizeQTable.MaxQ"/>).</param>
public readonly record struct SizeQPoint(float DaysFromPerihelion, float Q);

/// <summary>
/// The size law exponent q along the orbit ("Orbit-dependent size distribution"): up to
/// <see cref="MaxRows"/> rows of (days from perihelion, q), interpolated linearly in time and held
/// constant outside the table; one row is a constant q, no row means the jet's own q. Mirrors the
/// engine's <c>ParticleSystemEmitParams::size_q_table</c>. Immutable, with value equality.
/// </summary>
public sealed class SizeQTable : IEquatable<SizeQTable>
{
  /// <summary>The engine's table capacity (<c>SIZE_Q_TABLE_MAX</c>).</summary>
  public const int MaxRows = 8;
  /// <summary>Accepted q range of a row (<c>SIZE_Q_TABLE_RANGE</c>).</summary>
  public const float MinQ = 0f;
  public const float MaxQ = 6f;

  /// <summary>No rows: the jet's own q.</summary>
  public static SizeQTable Empty { get; } = new(ImmutableArray<SizeQPoint>.Empty);

  /// <summary>The rows, in the order given (the UI keeps them sorted by time).</summary>
  public ImmutableArray<SizeQPoint> Rows { get; }

  public int Count => Rows.Length;
  public bool IsEmpty => Rows.IsEmpty;

  private SizeQTable(ImmutableArray<SizeQPoint> rows) => Rows = rows;

  /// <summary>
  /// A table from <paramref name="rows"/>: a NaN time or q is dropped, q is clamped to the accepted
  /// range, the rows are sorted by time and the table cut to <see cref="MaxRows"/>.
  /// </summary>
  public static SizeQTable From(IEnumerable<SizeQPoint> rows)
  {
    var list = rows
      .Where(r => !float.IsNaN(r.DaysFromPerihelion) && !float.IsInfinity(r.DaysFromPerihelion) && !float.IsNaN(r.Q))
      .Select(r => new SizeQPoint(r.DaysFromPerihelion, Math.Min(MaxQ, Math.Max(MinQ, r.Q))))
      .OrderBy(r => r.DaysFromPerihelion)
      .Take(MaxRows)
      .ToImmutableArray();
    return list.IsEmpty ? Empty : new SizeQTable(list);
  }

  /// <summary>A table from parallel arrays (the savefile's form); null or mismatched lengths give <see cref="Empty"/>.</summary>
  public static SizeQTable From(float[]? days, float[]? q)
  {
    if (days is null || q is null || days.Length != q.Length)
      return Empty;
    var rows = new SizeQPoint[days.Length];
    for (int i = 0; i < rows.Length; i++)
      rows[i] = new SizeQPoint(days[i], q[i]);
    return From(rows);
  }

  /// <summary>
  /// q at <paramref name="daysFromPerihelion"/>: linear between rows, the end rows' values outside,
  /// <paramref name="fallbackQ"/> without rows (the engine's <c>size_q_at</c>).
  /// </summary>
  public float Evaluate(double daysFromPerihelion, float fallbackQ)
  {
    if (Rows.IsEmpty || double.IsNaN(daysFromPerihelion))
      return fallbackQ;
    float t = (float)daysFromPerihelion;
    if (t <= Rows[0].DaysFromPerihelion)
      return Rows[0].Q;
    var last = Rows[Rows.Length - 1];
    if (t >= last.DaysFromPerihelion)
      return last.Q;
    for (int i = 0; i + 1 < Rows.Length; i++)
    {
      var a = Rows[i];
      var b = Rows[i + 1];
      if (t >= a.DaysFromPerihelion && t <= b.DaysFromPerihelion)
      {
        float span = b.DaysFromPerihelion - a.DaysFromPerihelion;
        if (!(span > 0f))
          return b.Q;
        return a.Q + (b.Q - a.Q) * ((t - a.DaysFromPerihelion) / span);
      }
    }
    return last.Q;
  }

  /// <summary>The rows' times (days), for the savefile.</summary>
  public float[] DaysArray() => Rows.Select(r => r.DaysFromPerihelion).ToArray();

  /// <summary>The rows' q values, for the savefile.</summary>
  public float[] QArray() => Rows.Select(r => r.Q).ToArray();

  public bool Equals(SizeQTable? other) =>
    other is not null && (ReferenceEquals(this, other) || Rows.SequenceEqual(other.Rows));

  public override bool Equals(object? obj) => Equals(obj as SizeQTable);

  public override int GetHashCode()
  {
    unchecked
    {
      int h = 17;
      foreach (var r in Rows)
        h = h * 31 + r.GetHashCode();
      return h;
    }
  }

  public override string ToString() =>
    "[" + string.Join(", ", Rows.Select(r => string.Format(System.Globalization.CultureInfo.InvariantCulture,
      "({0} d, q {1})", r.DaysFromPerihelion, r.Q))) + "]";
}

/// <summary>
/// The dust lifetime (per jet) and what it implies: the engine emits each age tier on its own grid
/// of <see cref="WindowsPerTtl"/> windows over the tier's age band, so the emission step of a tier
/// is its band length / 256.
/// </summary>
public static class DustLifetime
{
  /// <summary>The engine's default and the value every older save used (<c>DEFAULT_TTL_DAYS</c>).</summary>
  public const float DefaultDays = 30f;
  /// <summary>Accepted range (<c>TTL_DAYS_RANGE</c>).</summary>
  public const float MinDays = 1f;
  public const float MaxDays = 365f;
  /// <summary>Emission windows per tier (<c>dust::WINDOWS_PER_TTL</c>).</summary>
  public const int WindowsPerTtl = 256;
  /// <summary>Age bands of the three tiers, as multiples of the lifetime (<c>dust::dust_tier_bands</c>).</summary>
  public static readonly (double MinTtl, double MaxTtl)[] TierBands = { (0, 1), (1, 8), (8, 64) };

  /// <summary>The lifetime in force for a typed value: within range, else the default (the engine's rule).</summary>
  public static float Effective(float days) =>
    float.IsNaN(days) || float.IsInfinity(days) || days < MinDays || days > MaxDays ? DefaultDays : days;

  /// <summary>The emission step of each tier in days: (band length in lifetimes) · lifetime / 256.</summary>
  public static double[] EmissionStepsDays(float lifetimeDays)
  {
    double ttl = Effective(lifetimeDays);
    var steps = new double[TierBands.Length];
    for (int i = 0; i < steps.Length; i++)
      steps[i] = (TierBands[i].MaxTtl - TierBands[i].MinTtl) * ttl / WindowsPerTtl;
    return steps;
  }

  /// <summary>A step as "2.8 h" below a day, "6.6 d" from a day on.</summary>
  public static string FormatStep(double days)
  {
    var ci = System.Globalization.CultureInfo.InvariantCulture;
    if (days < 1.0)
      return (days * 24.0).ToString("0.0", ci) + " h";
    return days.ToString("0.0", ci) + " d";
  }

  /// <summary>The three steps joined: "2.8 h / 19.7 h / 6.6 d" at 30 days.</summary>
  public static string EmissionStepsText(float lifetimeDays) =>
    string.Join(" / ", EmissionStepsDays(lifetimeDays).Select(FormatStep));
}
