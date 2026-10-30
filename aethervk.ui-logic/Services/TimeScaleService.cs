using System;
using System.Globalization;
using AetherVk.Logic.Utils;

namespace AetherVk.Logic.Services;

/// <summary>The calendar scale dates are typed and shown in (Settings › Time scale for dates).</summary>
public enum TimeScale
{
  /// <summary>Coordinated Universal Time: what a telescope image's DATE-OBS uses. The default.</summary>
  Utc,
  /// <summary>Barycentric Dynamical Time (ET): the engine's own scale, TDB = UTC + 69.18 s since 2017.</summary>
  Tdb,
}

/// <summary>
/// The chosen <see cref="TimeScale"/> (saved in preferences.json) and the text ↔ engine-epoch
/// conversions in it. The engine, the committed range and the save file always hold TDB parts;
/// only typed and displayed texts are in the chosen scale, through <see cref="ITimeScaleConverter"/>
/// (the core's leap-second table). Without a converter (tests, design time) the service is
/// TDB-only: texts are read as TDB, as before this round.
/// </summary>
public sealed class TimeScaleService
{
  public const string UtcName = "utc";
  public const string TdbName = "tdb";

  /// <summary>The box format of the Comet tab's range boxes.</summary>
  public const string IsoSecondsFormat = "yyyy-MM-dd'T'HH:mm:ss";

  private readonly PreferencesService? _preferences;
  private TimeScale _scale;

  /// <summary>The converter; null: TDB only.</summary>
  public ITimeScaleConverter? Converter { get; }

  /// <summary>Today (UTC), for the leap-second validity horizon; replaced in tests.</summary>
  public Func<DateTime> Now { get; set; } = () => DateTime.UtcNow;

  /// <param name="converter">the UTC ↔ TDB conversion; null: TDB only</param>
  /// <param name="preferences">the saved scale (read at construction, written by <see cref="Scale"/>)</param>
  public TimeScaleService(ITimeScaleConverter? converter, PreferencesService? preferences = null)
  {
    Converter = converter;
    _preferences = preferences;
    _scale = converter is null ? TimeScale.Tdb : preferences?.TimeScale ?? TimeScale.Utc;
  }

  /// <summary>A service without a converter: texts are TDB (tests of code that takes none).</summary>
  public static TimeScaleService TdbOnly() => new(null);

  /// <summary>The chosen scale; forced to TDB without a converter. Writes the preference.</summary>
  public TimeScale Scale
  {
    get => _scale;
    set
    {
      if (Converter is null) value = TimeScale.Tdb;
      if (value == _scale) return;
      _scale = value;
      _preferences?.SetTimeScale(value);
      ScaleChanged?.Invoke(value);
    }
  }

  /// <summary>Raised after <see cref="Scale"/> changed (the boxes reformat, the labels follow).</summary>
  public event Action<TimeScale>? ScaleChanged;

  public bool IsUtc => _scale == TimeScale.Utc;

  /// <summary>"UTC" or "TDB" (the labels' suffix).</summary>
  public string ScaleName => NameOf(_scale);

  public static string NameOf(TimeScale scale) => scale == TimeScale.Utc ? "UTC" : "TDB";

  /// <summary>The preference / save-file name: "utc" or "tdb".</summary>
  public static string KeyOf(TimeScale scale) => scale == TimeScale.Utc ? UtcName : TdbName;

  /// <summary>Parses "utc" / "tdb" (any case); anything else is <paramref name="fallback"/>.</summary>
  public static TimeScale Parse(string? key, TimeScale fallback)
  {
    if (string.Equals(key, UtcName, StringComparison.OrdinalIgnoreCase)) return TimeScale.Utc;
    if (string.Equals(key, TdbName, StringComparison.OrdinalIgnoreCase)) return TimeScale.Tdb;
    return fallback;
  }

  /// <summary>The scale a session's texts are in: its pinned scale, else the current one.</summary>
  public TimeScale ScaleOf(TimelineSession s) => Parse(s.ProposedScale, _scale);

  // ── text → parts ───────────────────────────────────────────────────────────

  /// <summary>Parses <paramref name="text"/> in the current scale to TDB parts.</summary>
  public bool TryParse(string text, out short centuries, out ulong nanoseconds, out UtcStatus status) =>
    TryParse(text, _scale, out centuries, out nanoseconds, out status);

  /// <summary>
  /// Parses an ISO-8601 <paramref name="text"/> read in <paramref name="scale"/> to TDB parts.
  /// False when it is no date, or a UTC date without a converter (<see cref="UtcStatus.Malformed"/>:
  /// never read as TDB, 69 s off); <paramref name="status"/> reports a UTC date outside the
  /// leap-second table (the value is still converted: the date is never refused).
  /// </summary>
  public bool TryParse(string text, TimeScale scale, out short centuries, out ulong nanoseconds, out UtcStatus status)
  {
    status = UtcStatus.Ok;
    centuries = 0;
    nanoseconds = 0;
    if (!TimeUtils.TryParseIso8601(text, out var dt))
      return false;
    if (scale == TimeScale.Tdb)
    {
      (centuries, nanoseconds) = TimeUtils.ToTdbParts(dt);
      return true;
    }
    if (Converter is null)
    {
      status = UtcStatus.Malformed;
      return false;
    }
    status = Converter.UtcToTdbParts(dt, out centuries, out nanoseconds);
    return status != UtcStatus.Malformed;
  }

  /// <summary>Parses in the current scale; false when no date.</summary>
  public bool TryParse(string text, out (short Centuries, ulong Nanoseconds) parts)
  {
    bool ok = TryParse(text, out short c, out ulong ns, out _);
    parts = (c, ns);
    return ok;
  }

  /// <summary>
  /// UTC instant → TDB parts (the FITS observation epoch). Throws without a converter or when the
  /// core refuses the instant: taking the UTC digits as TDB would put it 69 s off.
  /// </summary>
  public (short Centuries, ulong Nanoseconds) FromUtc(DateTimeOffset utc)
  {
    if (Converter is null)
      throw new InvalidOperationException($"UTC {utc:O} → TDB needs the leap-second converter (a TDB-only service)");
    if (Converter.UtcToTdbParts(utc, out short c, out ulong ns) == UtcStatus.Malformed)
      throw new ArgumentOutOfRangeException(nameof(utc), utc, "the leap-second table refused this UTC instant");
    return (c, ns);
  }

  // ── parts → text ───────────────────────────────────────────────────────────

  /// <summary><paramref name="format"/> of the TDB parts in the current scale.</summary>
  public string Format(short centuries, ulong nanoseconds, string format) => Format(centuries, nanoseconds, format, _scale);

  /// <summary>
  /// <paramref name="format"/> (invariant) of the TDB parts shown in <paramref name="scale"/>. A UTC
  /// text the converter cannot give (none, or refused) is the TDB text labelled " TDB", never
  /// passed off as UTC.
  /// </summary>
  public string Format(short centuries, ulong nanoseconds, string format, TimeScale scale)
  {
    // the simulation clock counts whole µs from the start: a seek to a whole second can land up
    // to 1 µs short of it, which a text without a fraction showed as the second before
    // ("2021-12-14 00:00:00" read back as 23:59:59). Such a text shows the instant half a ms
    // later (the nearest ms in the shown scale, truncated). Rounding the TDB parts is not enough:
    // TDB - UTC has a sub-ms part (69.1834 s), so a TDB rounded to .183 is UTC 0.4 ms short.
    if (format.IndexOf('f') < 0 && format.IndexOf('F') < 0)
      (centuries, nanoseconds) = HalfMsLater(centuries, nanoseconds);
    if (scale == TimeScale.Tdb)
      return TimeUtils.FormatTdbEpoch(centuries, nanoseconds, format);
    if (Converter is null || Converter.TdbPartsToUtc(centuries, nanoseconds, out UtcFields f) == UtcStatus.Malformed)
      return TimeUtils.FormatTdbEpoch(centuries, nanoseconds, format) + " TDB";
    return f.ToString(format);
  }

  private const ulong NanosecondsPerCentury = 36525UL * 86400UL * 1_000_000_000UL;

  /// <summary>The parts 0.5 ms later (carried into the next century).</summary>
  internal static (short Centuries, ulong Nanoseconds) HalfMsLater(short centuries, ulong nanoseconds)
  {
    ulong ns = nanoseconds + 500_000UL;
    return ns >= NanosecondsPerCentury ? ((short)(centuries + 1), ns - NanosecondsPerCentury) : (centuries, ns);
  }

  /// <summary>
  /// The text for the epoch: <paramref name="sameInstant"/> when it names the same instant in
  /// <paramref name="scale"/> (to 1 ms; keeps what the user typed), else ISO-8601 seconds with a Z.
  /// </summary>
  public string EpochText(short centuries, ulong nanoseconds, string sameInstant, TimeScale scale)
  {
    if (TryParse(sameInstant, scale, out short c, out ulong ns, out _) && Math.Abs(SecondsBetween(c, ns, centuries, nanoseconds)) < 1e-3)
      return sameInstant;
    return Format(centuries, nanoseconds, IsoSecondsFormat, scale) + "Z";
  }

  /// <inheritdoc cref="EpochText(short, ulong, string, TimeScale)"/>
  public string EpochText(short centuries, ulong nanoseconds, string sameInstant) =>
    EpochText(centuries, nanoseconds, sameInstant, _scale);

  /// <summary>
  /// <paramref name="text"/> rewritten from <paramref name="from"/> to <paramref name="to"/>:
  /// the same instant, ISO-8601 to the second (with the fraction when there is one, with a Z when
  /// the text had one). Unparsable text, or the same scale, is returned as it is.
  /// </summary>
  public string Reformat(string text, TimeScale from, TimeScale to)
  {
    if (from == to || Converter is null || !TryParse(text, from, out short c, out ulong ns, out _))
      return text;
    string fmt = ns % 1_000_000_000UL == 0 ? IsoSecondsFormat : IsoSecondsFormat + ".FFFFFFF";
    string s = Format(c, ns, fmt, to);
    return text.TrimEnd().EndsWith("Z", StringComparison.Ordinal) ? s + "Z" : s;
  }

  /// <summary>
  /// Pins <paramref name="s"/>'s texts (proposed and committed range) to the current scale:
  /// rewritten from the scale they are in when it differs. The committed parts do not change.
  /// Returns true when a text was rewritten. Idempotent.
  /// </summary>
  public bool ApplyTo(TimelineSession s)
  {
    var from = ScaleOf(s);
    if (from == _scale && !string.IsNullOrEmpty(s.ProposedScale))
      return false;
    s.ProposedScale = KeyOf(_scale);
    if (from == _scale)
      return false;
    s.ProposedStartEpoch = Reformat(s.ProposedStartEpoch, from, _scale);
    s.ProposedEndEpoch = Reformat(s.ProposedEndEpoch, from, _scale);
    s.CommittedStartEpoch = Reformat(s.CommittedStartEpoch, from, _scale);
    s.CommittedEndEpoch = Reformat(s.CommittedEndEpoch, from, _scale);
    return true;
  }

  // ── status, help ───────────────────────────────────────────────────────────

  /// <summary>The leap-second status of a text typed in the current scale (TDB: always Ok).</summary>
  public UtcStatus StatusOf(string text)
  {
    TryParse(text, out _, out _, out var status);
    return _scale == TimeScale.Tdb ? UtcStatus.Ok : status;
  }

  /// <summary>The leap-second status of the epoch shown in the current scale (TDB: always Ok).</summary>
  public UtcStatus StatusOf(short centuries, ulong nanoseconds)
  {
    if (_scale == TimeScale.Tdb || Converter is null) return UtcStatus.Ok;
    return Converter.TdbPartsToUtc(centuries, nanoseconds, out UtcFields _);
  }

  /// <summary>A status the boxes warn about (the date is converted all the same).</summary>
  public static bool Warns(UtcStatus status) => status is UtcStatus.BeyondLskValidity or UtcStatus.BeforeLeapSeconds;

  /// <summary>"TDB = UTC + 69.18 s" at the epoch (at today's leap-second count without one).</summary>
  public string OffsetText(short? centuries = null, ulong? nanoseconds = null)
  {
    if (Converter is null) return string.Empty;
    short c;
    ulong ns;
    if (centuries is { } cc && nanoseconds is { } nn) (c, ns) = (cc, nn);
    else (c, ns) = TimeUtils.ToTdbParts(new DateTimeOffset(Now(), TimeSpan.Zero));
    double off = Converter.TdbMinusUtcSeconds(c, ns);
    return double.IsNaN(off) ? string.Empty : string.Format(CultureInfo.InvariantCulture, "TDB = UTC + {0:0.00} s", off);
  }

  /// <summary>The label tooltip: which scale the dates are in and the offset to the other.</summary>
  public string ScaleTip
  {
    get
    {
      string off = OffsetText();
      string tail = off.Length > 0 ? " " + off + "." : string.Empty;
      return IsUtc
        ? "Dates are UTC (what a telescope image's DATE-OBS uses)." + tail + " Settings › Time scale for dates."
        : "Dates are TDB (the engine's dynamical time, ET)." + tail + " Settings › Time scale for dates.";
    }
  }

  /// <summary>"Leap seconds: naif0012.tls, last +37 s on 2017-01-01" (empty without a converter).</summary>
  public string LeapSecondsSummary => Converter is null ? string.Empty : "Leap seconds: " + Converter.LeapSecondsInfo().Summary;

  /// <summary>The last date a UTC conversion is exact: the table's last entry (or today) + 6 months.</summary>
  public DateTime? ValidityEnd => Converter?.LeapSecondsInfo().ValidityEnd(Now());

  /// <summary>The Settings help: the two scales and the active leap-second table.</summary>
  public string HelpText
  {
    get
    {
      string off = OffsetText();
      return "UTC is civil time, what a telescope image's DATE-OBS and an ephemeris query use; it carries leap seconds."
        + "\n\nTDB (Barycentric Dynamical Time, ET) is the uniform time the engine and the SPK ephemerides run on."
        + (off.Length > 0 ? " Today " + off + "." : string.Empty)
        + "\n\nThe committed range, the save file and the engine always hold TDB; this choice only sets how dates are typed and shown."
        + (LeapSecondsSummary.Length > 0 ? "\n\n" + LeapSecondsSummary + "." : string.Empty);
    }
  }

  /// <summary>The warning next to a box whose UTC date is outside the leap-second table (null when none).</summary>
  public string? WarningText(UtcStatus status)
  {
    switch (status)
    {
      case UtcStatus.BeyondLskValidity:
        return string.Format(CultureInfo.InvariantCulture,
          "A UTC date after {0:yyyy-MM-dd} may be off by a leap second announced later (the leap-second table ends {1:yyyy-MM-dd}: {2}). TDB/ET is exact. The date is converted all the same; switch in Settings › Time scale for dates.",
          ValidityEnd ?? Now(), Converter?.LeapSecondsInfo().LastEntry ?? Now(), Converter?.LeapSecondsInfo().Summary ?? string.Empty);
      case UtcStatus.BeforeLeapSeconds:
        return "Before 1972 UTC had no leap seconds: the date is converted with a nominal TAI − UTC of 10 s. TDB/ET is exact; switch in Settings › Time scale for dates.";
      default:
        return null;
    }
  }

  private static double SecondsBetween(short c1, ulong n1, short c2, ulong n2) =>
    (c1 - c2) * 36525.0 * 86400.0 + ((double)n1 - (double)n2) * 1e-9;
}
