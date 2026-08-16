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
}
