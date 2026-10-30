using System;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;

namespace AetherVk.Logic.Tests.Mocks;

/// <summary>
/// A deterministic <see cref="ITimeScaleConverter"/>: TDB = UTC + 69.184 s (ΔAT 37 s + 32.184 s,
/// no periodic term, no leap second inside a range), a leap-second table ending 2017-01-01 and a
/// fixed "today", so the validity horizon (<see cref="ValidityEnd"/>) is known to the tests.
/// </summary>
public sealed class FakeTimeScaleConverter : ITimeScaleConverter
{
  public const double OffsetSeconds = 69.184;
  /// <summary>The offset in ticks (100 ns): integer-exact, no double rounding (69.184 s × 10⁷ truncates to …9999).</summary>
  public static readonly TimeSpan Offset = TimeSpan.FromTicks(691_840_000);

  public DateTime LastEntry { get; set; } = new(2017, 1, 1, 0, 0, 0, DateTimeKind.Utc);
  public DateTime Now { get; set; } = new(2026, 10, 30, 0, 0, 0, DateTimeKind.Utc);
  public string Path { get; set; } = "/assets/planets/naif0012.tls";

  /// <summary>Last UTC instant converted exactly: max(last entry, now) + 6 months.</summary>
  public DateTime ValidityEnd => LeapSecondsInfo().ValidityEnd(Now);

  public int UtcToTdbCalls { get; private set; }

  public static (short Centuries, ulong Nanoseconds) Parts(DateTimeOffset utc) =>
    TimeUtils.ToTdbParts(utc + Offset);

  public UtcStatus UtcToTdbParts(UtcFields utc, out short centuries, out ulong nanoseconds)
  {
    UtcToTdbCalls++;
    centuries = 0;
    nanoseconds = 0;
    if (utc.Month is < 1 or > 12 || utc.Day < 1 || utc.Hour > 23 || utc.Minute > 59 || utc.Second > 60)
      return UtcStatus.Malformed;
    if (utc.Second == 60 && !(utc.Year == 2016 && utc.Month == 12 && utc.Day == 31 && utc.Hour == 23 && utc.Minute == 59))
      return UtcStatus.Malformed;
    DateTimeOffset dt;
    try { dt = utc.ToDateTimeOffset(); }
    catch (ArgumentOutOfRangeException) { return UtcStatus.Malformed; }
    (centuries, nanoseconds) = Parts(dt);
    return StatusOf(dt);
  }

  public UtcStatus TdbPartsToUtc(short centuries, ulong nanoseconds, out UtcFields utc)
  {
    var dt = TimeUtils.FromTdbParts(centuries, nanoseconds) - Offset;
    utc = UtcFields.From(dt);
    return StatusOf(dt);
  }

  private UtcStatus StatusOf(DateTimeOffset utc)
  {
    if (utc.Year < 1972) return UtcStatus.BeforeLeapSeconds;
    if (utc.UtcDateTime > ValidityEnd) return UtcStatus.BeyondLskValidity;
    return UtcStatus.Ok;
  }

  public LeapSecondsInfo LeapSecondsInfo() => new(true, 28, 37, LastEntry, Path);
}
