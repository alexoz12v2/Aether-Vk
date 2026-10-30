using System;
using System.Globalization;

namespace AetherVk.Logic.Services;

/// <summary>Result of a UTC ↔ TDB conversion (<c>leap_seconds::UtcStatus</c> in the core).</summary>
public enum UtcStatus
{
  Ok = 0,
  /// <summary>Before 1972: UTC has no leap-second definition; converted with a nominal ΔAT of 10 s.</summary>
  BeforeLeapSeconds = 1,
  /// <summary>Later than the leap-second table's validity horizon (its last entry or today, plus six
  /// months): converted assuming no further leap second.</summary>
  BeyondLskValidity = 2,
  /// <summary>Nothing converted (an impossible calendar date, or second 60 off a leap second).</summary>
  Malformed = 3,
}

/// <summary>Result of offering a NAIF leap-second kernel to the core (<c>leap_seconds::LoadStatus</c>).</summary>
public enum LeapSecondsLoadStatus
{
  Adopted = 0,
  KeptCurrent = 1,
  Unreadable = 2,
  Malformed = 3,
  Inconsistent = 4,
}

/// <summary>Calendar fields of a UTC instant; <see cref="Second"/> is 60 on a leap second.</summary>
public readonly struct UtcFields : IEquatable<UtcFields>
{
  public int Year { get; }
  public int Month { get; }
  public int Day { get; }
  public int Hour { get; }
  public int Minute { get; }
  public int Second { get; }
  public uint Nanos { get; }

  public UtcFields(int year, int month, int day, int hour, int minute, int second, uint nanos = 0)
  {
    Year = year;
    Month = month;
    Day = day;
    Hour = hour;
    Minute = minute;
    Second = second;
    Nanos = nanos;
  }

  public static UtcFields From(DateTimeOffset utc)
  {
    var u = utc.UtcDateTime;
    return new UtcFields(u.Year, u.Month, u.Day, u.Hour, u.Minute, u.Second, (uint)(u.Ticks % TimeSpan.TicksPerSecond) * 100u);
  }

  /// <summary>
  /// The fields as a <see cref="DateTimeOffset"/>; a leap second (second 60) becomes the last
  /// 100 ns tick of 23:59:59, which .NET can hold.
  /// </summary>
  public DateTimeOffset ToDateTimeOffset()
  {
    if (Second >= 60)
      return new DateTimeOffset(Year, Month, Day, Hour, Minute, 59, TimeSpan.Zero)
        .AddTicks(TimeSpan.TicksPerSecond - 1);
    return new DateTimeOffset(Year, Month, Day, Hour, Minute, Second, TimeSpan.Zero).AddTicks(Nanos / 100);
  }

  /// <summary>The fields in <paramref name="format"/> (invariant); second 60 is printed as such.</summary>
  public string ToString(string format)
  {
    if (Second < 60)
      return ToDateTimeOffset().ToString(format, CultureInfo.InvariantCulture);
    // the format applied to :59, then the seconds replaced
    string s = new DateTimeOffset(Year, Month, Day, Hour, Minute, 59, TimeSpan.Zero).ToString(format, CultureInfo.InvariantCulture);
    int i = s.LastIndexOf(":59", StringComparison.Ordinal);
    return i < 0 ? s : s.Substring(0, i) + ":60" + s.Substring(i + 3);
  }

  public override string ToString() => ToString("yyyy-MM-dd'T'HH:mm:ss");

  public bool Equals(UtcFields o) =>
    Year == o.Year && Month == o.Month && Day == o.Day && Hour == o.Hour && Minute == o.Minute && Second == o.Second && Nanos == o.Nanos;

  public override bool Equals(object? obj) => obj is UtcFields o && Equals(o);

  public override int GetHashCode() => (Year * 397 ^ Month * 31 ^ Day) * 397 ^ (Hour * 3600 + Minute * 60 + Second) ^ (int)Nanos;
}

/// <summary>The active leap-second table (<c>avkTime_leapSecondsInfo</c>).</summary>
/// <param name="FromFile">true: a NAIF LSK file at <paramref name="Path"/>; false: the built-in table.</param>
/// <param name="EntryCount">entries in the table (28 for naif0012.tls).</param>
/// <param name="LastDeltaAt">TAI − UTC after the last entry (37 since 2017-01-01).</param>
/// <param name="LastEntry">UTC date of the last entry.</param>
/// <param name="Path">the LSK path; empty for the built-in table.</param>
public sealed record LeapSecondsInfo(bool FromFile, int EntryCount, int LastDeltaAt, DateTime LastEntry, string Path)
{
  /// <summary>Months after the last entry (or today) during which no new leap second can be announced
  /// unseen (IERS Bulletin C is published six months ahead).</summary>
  public const int ValidityMonthsAfterLastEntry = 6;

  /// <summary>The table's validity horizon: max(last entry, <paramref name="now"/>) + 6 months.</summary>
  public DateTime ValidityEnd(DateTime now)
  {
    var last = LastEntry > now ? LastEntry : now;
    return last.AddMonths(ValidityMonthsAfterLastEntry);
  }

  /// <summary>"naif0012.tls, last +37 s on 2017-01-01" / "built-in table, last +37 s on 2017-01-01".</summary>
  public string Summary =>
    string.Format(CultureInfo.InvariantCulture, "{0}, last {1:+#;-#;0} s on {2:yyyy-MM-dd}",
      FromFile ? System.IO.Path.GetFileName(Path) : "built-in table", LastDeltaAt, LastEntry);
}

/// <summary>
/// The one UTC ↔ TDB conversion point for the UI (the core's hifitime + leap-second table, or a
/// deterministic fake in tests). Engine epochs are TDB parts (centuries, nanoseconds since J2000);
/// a calendar date typed or shown as UTC goes through here.
/// </summary>
public interface ITimeScaleConverter
{
  /// <summary>UTC calendar fields → TDB parts; <see cref="UtcStatus.Malformed"/> writes zeros.</summary>
  UtcStatus UtcToTdbParts(UtcFields utc, out short centuries, out ulong nanoseconds);

  /// <summary>TDB parts → UTC calendar fields (second 60 on a leap second).</summary>
  UtcStatus TdbPartsToUtc(short centuries, ulong nanoseconds, out UtcFields utc);

  /// <summary>The active leap-second table.</summary>
  LeapSecondsInfo LeapSecondsInfo();
}

public static class TimeScaleConverterExtensions
{
  /// <summary>UTC instant → TDB parts.</summary>
  public static UtcStatus UtcToTdbParts(this ITimeScaleConverter c, DateTimeOffset utc, out short centuries, out ulong nanoseconds) =>
    c.UtcToTdbParts(UtcFields.From(utc), out centuries, out nanoseconds);

  /// <summary>TDB parts → UTC instant (a leap second lands on the last tick of :59).</summary>
  public static UtcStatus TdbPartsToUtc(this ITimeScaleConverter c, short centuries, ulong nanoseconds, out DateTimeOffset utc)
  {
    var status = c.TdbPartsToUtc(centuries, nanoseconds, out var f);
    utc = status == UtcStatus.Malformed ? default : f.ToDateTimeOffset();
    return status;
  }

  /// <summary>TDB − UTC at the given epoch, seconds (≈ 69.18 s in 2017–2026).</summary>
  public static double TdbMinusUtcSeconds(this ITimeScaleConverter c, short centuries, ulong nanoseconds)
  {
    if (c.TdbPartsToUtc(centuries, nanoseconds, out DateTimeOffset utc) == UtcStatus.Malformed)
      return double.NaN;
    return (Utils.TimeUtils.FromTdbParts(centuries, nanoseconds) - utc).TotalSeconds;
  }
}

/// <summary>The converter over the core's FFI (<c>avkTime_*</c>, <see cref="INativeRuntimeService"/>).</summary>
public sealed class NativeTimeScaleConverter : ITimeScaleConverter
{
  private readonly INativeRuntimeService _runtime;

  public NativeTimeScaleConverter(INativeRuntimeService runtime)
  {
    _runtime = runtime;
  }

  public UtcStatus UtcToTdbParts(UtcFields utc, out short centuries, out ulong nanoseconds) =>
    _runtime.UtcToTdb(utc, out centuries, out nanoseconds);

  public UtcStatus TdbPartsToUtc(short centuries, ulong nanoseconds, out UtcFields utc) =>
    _runtime.TdbToUtc(centuries, nanoseconds, out utc);

  public LeapSecondsInfo LeapSecondsInfo() => _runtime.LeapSecondsInfo();
}
