using System;
using System.Collections.Generic;
using System.Text.Json;

namespace AetherVk.Logic.Models;

/// <summary>How a FITS texture maps data values to grey (<c>avkSimulationContext_setFitsStretch</c> mode).</summary>
public enum FitsStretchMode
{
  /// <summary>IRAF zscale (the import default); no parameters.</summary>
  ZScale = 0,

  /// <summary>Black / white at the low / high percentiles of the data (P0 / P1 in 0..100).</summary>
  Percentile = 1,

  /// <summary>Black / white given in data units (P0 &gt; P1 renders inverted).</summary>
  Manual = 2,
}

/// <summary>Transfer curve applied between the black and white points.</summary>
public enum FitsTransfer
{
  Linear = 0,
  Log = 1,
  Asinh = 2,
}

/// <summary>A FITS stretch request. <see cref="Default"/> is what an import renders.</summary>
public sealed record FitsStretch(FitsStretchMode Mode, FitsTransfer Transfer, double P0, double P1)
{
  /// <summary>zscale, linear: the import's rendering.</summary>
  public static readonly FitsStretch Default = new(FitsStretchMode.ZScale, FitsTransfer.Linear, 0, 0);

  /// <summary>Same rendering (zscale ignores the parameters).</summary>
  public bool SameAs(FitsStretch? other)
  {
    other ??= Default;
    return Mode == other.Mode && Transfer == other.Transfer
      && (Mode == FitsStretchMode.ZScale || (P0 == other.P0 && P1 == other.P1));
  }
}

/// <summary>Black / white points (data units) a stretch used.</summary>
public sealed record FitsStretchResult(double Black, double White);

/// <summary>One header card: strings unquoted, HIERARCH keys without the prefix.</summary>
public sealed record FitsCard(string Key, string Value, string Comment);

/// <summary>
/// What the native parser read from the header (<c>docs/fits_ffi_contract.md</c>, "facts"); every
/// field is null when the header lacks it.
/// </summary>
public sealed record FitsFacts
{
  public string? Object { get; init; }
  public string? Telescope { get; init; }
  public string? Instrument { get; init; }
  public string? Filter { get; init; }
  public double? ExposureS { get; init; }

  /// <summary>Raw value of the date card used (<see cref="DateSource"/>).</summary>
  public string? DateObs { get; init; }

  public string? DateSource { get; init; }

  /// <summary>The card is already mid-exposure; else mid = start + EXPTIME / 2.</summary>
  public bool? DateIsMid { get; init; }

  public string? DateMidUtc { get; init; }

  /// <summary>Mid-exposure TDB as the <c>(centuries, nanoseconds)</c> parts the seek takes.</summary>
  public short? DateMidTdbCenturies { get; init; }

  public ulong? DateMidTdbNanoseconds { get; init; }

  /// <summary>No TIMESYS card: the date was read as UTC.</summary>
  public bool? TimesysAssumed { get; init; }

  public double? PixelScaleArcsec { get; init; }
  public string? PixelScaleSource { get; init; }
  public double? FovWidthArcmin { get; init; }
  public double? FovHeightArcmin { get; init; }

  /// <summary>Image centre through the TAN WCS (ICRS degrees).</summary>
  public double? CenterRaDeg { get; init; }

  public double? CenterDecDeg { get; init; }
  public double? CrvalRaDeg { get; init; }
  public double? CrvalDecDeg { get; init; }
  public double? PointingRaDeg { get; init; }
  public double? PointingDecDeg { get; init; }
  public string? Radesys { get; init; }

  /// <summary>Why the TAN WCS was refused (non-TAN, FK4, …).</summary>
  public string? WcsError { get; init; }

  /// <summary>Position angle of image +Y (texture up), degrees east of north, (−180, 180].</summary>
  public double? RotationDeg { get; init; }

  /// <summary>det CD &gt; 0: east is to the right with north up (a rotation cannot match it).</summary>
  public bool? Mirrored { get; init; }

  public double? SiteLatDeg { get; init; }

  /// <summary>East positive, after the LST check / west-convention rule.</summary>
  public double? SiteLonDeg { get; init; }

  public double? SiteElevM { get; init; }
  public string? SiteSource { get; init; }

  /// <summary>The value was negated because the card says "west".</summary>
  public bool? SiteLonWestConvention { get; init; }

  /// <summary>As written in the card.</summary>
  public double? SiteLonRawDeg { get; init; }

  /// <summary>The header's LST decided the sign (overrides the wording).</summary>
  public bool? SiteLonFromLst { get; init; }

  /// <summary>Where the image is centred on the sky: the WCS centre, else CRVAL, else the pointing.</summary>
  public (double RaDeg, double DecDeg)? SkyCenter =>
    CenterRaDeg is { } r && CenterDecDeg is { } d ? (r, d)
    : CrvalRaDeg is { } r2 && CrvalDecDeg is { } d2 ? (r2, d2)
    : PointingRaDeg is { } r3 && PointingDecDeg is { } d3 ? (r3, d3)
    : null;
}

/// <summary>
/// Metadata of a FITS texture asset (<c>avkSimulationContext_getFitsMetadataJson</c>): every header
/// card, the facts read from them and the stretch currently rendered.
/// </summary>
public sealed record FitsMetadata(
  ulong AssetId,
  int TextureWidth,
  int TextureHeight,
  bool Flipped,
  FitsStretch Stretch,
  FitsStretchResult? StretchPoints,
  IReadOnlyList<FitsCard> Cards,
  FitsFacts Facts)
{
  /// <summary>
  /// Parses the native JSON (<see cref="JsonDocument"/>: no reflection, AOT safe). Null when it
  /// is not valid JSON.
  /// </summary>
  public static FitsMetadata? Parse(string? json)
  {
    if (string.IsNullOrEmpty(json))
      return null;
    try
    {
      using var doc = JsonDocument.Parse(json!);
      var root = doc.RootElement;
      if (root.ValueKind != JsonValueKind.Object)
        return null;
      var stretch = FitsStretch.Default;
      FitsStretchResult? points = null;
      if (Obj(root, "stretch") is { } s)
      {
        var mode = (FitsStretchMode)(int)(Num(s, "mode") ?? 0);
        var transfer = (FitsTransfer)(int)(Num(s, "transfer") ?? 0);
        if (!Enum.IsDefined(typeof(FitsStretchMode), mode)) mode = FitsStretchMode.ZScale;
        if (!Enum.IsDefined(typeof(FitsTransfer), transfer)) transfer = FitsTransfer.Linear;
        double? black = Num(s, "black"), white = Num(s, "white");
        // the parameters behind black / white are not echoed: a percentile keeps none
        stretch = new FitsStretch(mode, transfer,
          mode == FitsStretchMode.Manual ? black ?? 0 : 0, mode == FitsStretchMode.Manual ? white ?? 0 : 0);
        if (black is { } b && white is { } w)
          points = new FitsStretchResult(b, w);
      }
      var cards = new List<FitsCard>();
      if (root.TryGetProperty("cards", out var arr) && arr.ValueKind == JsonValueKind.Array)
        foreach (var c in arr.EnumerateArray())
          if (c.ValueKind == JsonValueKind.Object)
            cards.Add(new FitsCard(Str(c, "key") ?? "", Str(c, "value") ?? "", Str(c, "comment") ?? ""));
      var facts = Obj(root, "facts") is { } f ? ParseFacts(f) : new FitsFacts();
      return new FitsMetadata(
        (ulong)(Num(root, "assetId") ?? 0),
        (int)(Num(root, "textureWidth") ?? 0),
        (int)(Num(root, "textureHeight") ?? 0),
        Bool(root, "flipped") ?? false,
        stretch,
        points,
        cards,
        facts);
    }
    catch (JsonException)
    {
      return null;
    }
  }

  private static FitsFacts ParseFacts(JsonElement f) => new()
  {
    Object = Str(f, "object"),
    Telescope = Str(f, "telescope"),
    Instrument = Str(f, "instrument"),
    Filter = Str(f, "filter"),
    ExposureS = Num(f, "exposureS"),
    DateObs = Str(f, "dateObs"),
    DateSource = Str(f, "dateSource"),
    DateIsMid = Bool(f, "dateIsMid"),
    DateMidUtc = Str(f, "dateMidUtc"),
    DateMidTdbCenturies = Num(f, "dateMidTdbCenturies") is { } c ? (short)c : null,
    // u64: may exceed 2^53, read it exactly
    DateMidTdbNanoseconds = f.TryGetProperty("dateMidTdbNanoseconds", out var ns)
      && ns.ValueKind == JsonValueKind.Number && ns.TryGetUInt64(out ulong n) ? n : null,
    TimesysAssumed = Bool(f, "timesysAssumed"),
    PixelScaleArcsec = Num(f, "pixelScaleArcsec"),
    PixelScaleSource = Str(f, "pixelScaleSource"),
    FovWidthArcmin = Num(f, "fovWidthArcmin"),
    FovHeightArcmin = Num(f, "fovHeightArcmin"),
    CenterRaDeg = Num(f, "centerRaDeg"),
    CenterDecDeg = Num(f, "centerDecDeg"),
    CrvalRaDeg = Num(f, "crvalRaDeg"),
    CrvalDecDeg = Num(f, "crvalDecDeg"),
    PointingRaDeg = Num(f, "pointingRaDeg"),
    PointingDecDeg = Num(f, "pointingDecDeg"),
    Radesys = Str(f, "radesys"),
    WcsError = Str(f, "wcsError"),
    RotationDeg = Num(f, "rotationDeg"),
    Mirrored = Bool(f, "mirrored"),
    SiteLatDeg = Num(f, "siteLatDeg"),
    SiteLonDeg = Num(f, "siteLonDeg"),
    SiteElevM = Num(f, "siteElevM"),
    SiteSource = Str(f, "siteSource"),
    SiteLonWestConvention = Bool(f, "siteLonWestConvention"),
    SiteLonRawDeg = Num(f, "siteLonRawDeg"),
    SiteLonFromLst = Bool(f, "siteLonFromLst"),
  };

  private static JsonElement? Obj(JsonElement e, string name) =>
    e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Object ? v : null;

  private static string? Str(JsonElement e, string name) =>
    e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

  private static double? Num(JsonElement e, string name) =>
    e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number && v.TryGetDouble(out double d)
      && !double.IsNaN(d) && !double.IsInfinity(d)
      ? d
      : null;

  private static bool? Bool(JsonElement e, string name) =>
    e.TryGetProperty(name, out var v)
      ? v.ValueKind switch
      {
        JsonValueKind.True => true,
        JsonValueKind.False => false,
        _ => null,
      }
      : null;
}
