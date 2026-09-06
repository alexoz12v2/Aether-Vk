using System;

namespace AetherVk.Logic.Utils;

public static class TimeUtils
{
  private static readonly DateTimeOffset J2000 = new DateTimeOffset(2000, 1, 1, 12, 0, 0, TimeSpan.Zero);

  /// <summary>
  /// Parses an ISO 8601 string to a DateTimeOffset.
  /// </summary>
  public static bool TryParseIso8601(string isoString, out DateTimeOffset result)
  {
    return DateTimeOffset.TryParse(isoString, out result);
  }

  /// <summary>
  /// Converts a DateTimeOffset to TAI parts (Centuries and Nanoseconds relative to J2000).
  /// </summary>
  public static (short centuries, ulong nanoseconds) ToTaiParts(DateTimeOffset time)
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
  /// Converts TAI parts back to a DateTimeOffset.
  /// </summary>
  public static DateTimeOffset FromTaiParts(short centuries, ulong nanoseconds)
  {
    long ticksPerCentury = TimeSpan.TicksPerDay * 36525L;
    long ticks = (long)centuries * ticksPerCentury + (long)(nanoseconds / 100UL);
    return J2000.AddTicks(ticks);
  }

  /// <summary>
  /// Formats TAI parts as a UTC display string.
  /// </summary>
  public static string FormatTaiEpoch(short centuries, ulong nanoseconds)
  {
    return FromTaiParts(centuries, nanoseconds).ToString("yyyy-MM-dd HH:mm");
  }
}
