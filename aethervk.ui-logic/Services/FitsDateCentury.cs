using System;
using System.Globalization;
using AetherVk.Logic.Models;
using AetherVk.Logic.Utils;

namespace AetherVk.Logic.Services;

/// <summary>
/// Applies the user's century choice to the date facts of a FITS header whose date card wrote
/// its year with two digits (<see cref="FitsFacts.DateTwoDigitYear"/>): the engine reads
/// <c>'27/12/21'</c> as 2021; <see cref="FitsTwoDigitYearCentury.NineteenHundreds"/> moves the
/// mid-exposure and start instants back exactly 100 calendar years and recomputes their TDB
/// epochs from the shifted UTC through the app's UTC → TDB converter (no constant offset). The
/// raw card texts (<see cref="FitsFacts.DateObs"/>, <see cref="FitsFacts.DateCardRaw"/>) stay as
/// written.
/// </summary>
public static class FitsDateCentury
{
  private const long MicrosecondsPerCentury = 36525L * 86400L * 1_000_000L;

  /// <summary>The facts as the chosen century reads them; <paramref name="facts"/> itself when the
  /// header's year has four digits or the choice is the default 20yy.</summary>
  public static FitsFacts Apply(FitsFacts facts, FitsTwoDigitYearCentury century, ITimeScaleConverter? converter) =>
    century == FitsTwoDigitYearCentury.NineteenHundreds && facts.DateTwoDigitYear == true
      ? ShiftBackOneCentury(facts, converter)
      : facts;

  /// <summary>Every date of the facts 100 calendar years earlier, TDB recomputed from UTC.</summary>
  public static FitsFacts ShiftBackOneCentury(FitsFacts facts, ITimeScaleConverter? converter)
  {
    var mid = Shifted(facts.DateMidUtc);
    var start = Shifted(facts.DateStartUtc);
    var midParts = mid is { } m ? TdbParts(m, converter) : null;
    var startParts = start is { } st ? TdbParts(st, converter) : null;
    return facts with
    {
      DateMidUtc = mid is { } mu ? Iso(mu) : facts.DateMidUtc,
      DateStartUtc = start is { } su ? Iso(su) : facts.DateStartUtc,
      DateMidTdbCenturies = midParts?.Centuries ?? (mid is null ? facts.DateMidTdbCenturies : null),
      DateMidTdbNanoseconds = midParts?.Nanoseconds ?? (mid is null ? facts.DateMidTdbNanoseconds : null),
      DateStartTdbUs = startParts is { } sp ? Microseconds(sp.Centuries, sp.Nanoseconds)
        : start is null ? facts.DateStartTdbUs : null,
    };
  }

  /// <summary>The ISO UTC instant 100 calendar years earlier (29 February lands on 28 February
  /// when the earlier year is not a leap year); null when the text is not a date.</summary>
  private static DateTimeOffset? Shifted(string? isoUtc) =>
    isoUtc is { Length: > 0 } && TimeUtils.TryParseIso8601(isoUtc, out var t) ? t.AddYears(-100) : null;

  /// <summary>The engine's own format, <c>YYYY-MM-DDTHH:MM:SS.sssZ</c>.</summary>
  private static string Iso(DateTimeOffset utc) =>
    utc.UtcDateTime.ToString("yyyy-MM-dd'T'HH:mm:ss.fff'Z'", CultureInfo.InvariantCulture);

  /// <summary>UTC → TDB parts through <paramref name="converter"/> (the leap-second table; before
  /// 1972 a nominal ΔAT); without a converter (design time) the plain TDB-read difference. Null when
  /// the converter refuses the instant.</summary>
  private static (short Centuries, ulong Nanoseconds)? TdbParts(DateTimeOffset utc, ITimeScaleConverter? converter)
  {
    if (converter is null)
      return TimeUtils.ToTdbParts(utc);
    var status = converter.UtcToTdbParts(utc, out short centuries, out ulong nanoseconds);
    return status == UtcStatus.Malformed ? null : (centuries, nanoseconds);
  }

  /// <summary>TDB parts → microseconds since J2000 TDB (<c>FitsFacts.DateStartTdbUs</c>'s scale).</summary>
  public static long Microseconds(short centuries, ulong nanoseconds) =>
    centuries * MicrosecondsPerCentury + (long)(nanoseconds / 1000UL);
}
