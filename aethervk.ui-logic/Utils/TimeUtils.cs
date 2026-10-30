using System;
using System.Globalization;

namespace AetherVk.Logic.Utils;

public static class TimeUtils
{
  private static readonly DateTimeOffset J2000 = new DateTimeOffset(2000, 1, 1, 12, 0, 0, TimeSpan.Zero);

  /// <summary>
  /// Parses an ISO 8601 string to a DateTimeOffset. Culture-invariant, and a time without an
  /// offset is UTC: the OS culture and time zone used to decide both, so "2021-12-27 00:14:47"
  /// was an hour off on a UTC+1 machine and "12/27/2021"-style input depended on the locale.
  /// </summary>
  public static bool TryParseIso8601(string isoString, out DateTimeOffset result)
  {
    return DateTimeOffset.TryParse(isoString, CultureInfo.InvariantCulture,
      DateTimeStyles.AssumeUniversal | DateTimeStyles.AllowWhiteSpaces, out result);
  }

  /// <summary>
  /// Converts a calendar instant read as TDB to the engine's epoch parts (centuries and
  /// nanoseconds since J2000 TDB, <c>CTimeRange</c> / <c>seekEpochSync</c>): a plain difference
  /// from J2000, no leap seconds and no TT offset. A UTC instant goes through
  /// <see cref="Services.ITimeScaleConverter"/> instead (TDB = UTC + 69.18 s in 2017–2026).
  /// </summary>
  public static (short centuries, ulong nanoseconds) ToTdbParts(DateTimeOffset time)
  {
    var diff = time - J2000;

    long ticksPerDay = TimeSpan.TicksPerDay;
    long totalTicks = diff.Ticks;
    long ticksPerCentury = ticksPerDay * 36525L;

    long centuries = Math.DivRem(totalTicks, ticksPerCentury, out long remainderTicks);
    if (remainderTicks < 0)
    {
      centuries--;
      remainderTicks += ticksPerCentury;
    }

    ulong nanoseconds = (ulong)remainderTicks * 100UL;
    return ((short)centuries, nanoseconds);
  }

  /// <summary>
  /// The inverse of <see cref="ToTdbParts"/>: the engine epoch as a calendar instant read as TDB
  /// (a <see cref="DateTimeOffset"/> with a zero offset, not a UTC instant).
  /// </summary>
  public static DateTimeOffset FromTdbParts(short centuries, ulong nanoseconds)
  {
    long ticksPerCentury = TimeSpan.TicksPerDay * 36525L;
    long ticks = (long)centuries * ticksPerCentury + (long)(nanoseconds / 100UL);
    return J2000.AddTicks(ticks);
  }

  /// <summary>
  /// The text for the epoch (<paramref name="centuries"/>, <paramref name="nanoseconds"/>):
  /// <paramref name="sameInstant"/> when it names the same instant (to 1 ms; keeps what the user
  /// typed), else ISO-8601 with a Z. Texts read as TDB; <c>TimeScaleService.EpochText</c> for the
  /// chosen scale.
  /// </summary>
  public static string EpochText(short centuries, ulong nanoseconds, string sameInstant)
  {
    var at = FromTdbParts(centuries, nanoseconds);
    if (TryParseIso8601(sameInstant, out var p) && Math.Abs((p - at).TotalMilliseconds) < 1)
      return sameInstant;
    return at.UtcDateTime.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'", CultureInfo.InvariantCulture);
  }

  /// <summary>
  /// Formats engine epoch parts as a TDB calendar string ("yyyy-MM-dd HH:mm", invariant).
  /// </summary>
  public static string FormatTdbEpoch(short centuries, ulong nanoseconds)
  {
    return FromTdbParts(centuries, nanoseconds).ToString("yyyy-MM-dd HH:mm", CultureInfo.InvariantCulture);
  }

  /// <summary><paramref name="format"/> of the engine epoch parts read as TDB (invariant culture).</summary>
  public static string FormatTdbEpoch(short centuries, ulong nanoseconds, string format)
  {
    return FromTdbParts(centuries, nanoseconds).ToString(format, CultureInfo.InvariantCulture);
  }
}
