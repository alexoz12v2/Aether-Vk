using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.Utils;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

/// <summary>One line of the "Key facts" block.</summary>
public sealed record FitsFactItem(string Label, string Value);

/// <summary>
/// The "FITS image" part of a billboard row whose texture is a FITS asset: the key header facts,
/// every header card, the stretch, and the purpose buttons that set the simulation up like the
/// observation (seek, site, orientation, scale, comparison lock).
/// <para>Earth observer gating: "Orient to image" and "Match camera to image" need the Earth observer
/// view and are disabled outside it; "Seek to observation" and "Use observatory site" work in any
/// mode (the site is stored and used once the Earth observer is entered). "Set up comparison"
/// switches the viewport to the Earth observer itself, then runs every step it has the header facts
/// for.</para>
/// </summary>
public partial class FitsImageViewModel : ObservableObject
{
  private static readonly CultureInfo Ci = CultureInfo.InvariantCulture;

  private readonly DesktopBillboardItemViewModel _row;
  private bool _settingStretch;

  public ulong AssetId { get; }

  public string TextureKey { get; }

  public FitsMetadata Metadata { get; }

  public FitsFacts FactsData => Metadata.Facts;

  /// <summary>Every header card, in file order.</summary>
  public IReadOnlyList<FitsCard> Cards => Metadata.Cards;

  /// <summary>The key facts the header has (label, value), for the summary block.</summary>
  [ObservableProperty]
  private IReadOnlyList<FitsFactItem> _facts = [];

  /// <summary>The header has a site longitude (the flip toggle is shown).</summary>
  public bool HasSiteLon => FactsData.SiteLonDeg is not null;

  /// <summary>Use the opposite sign of the parsed longitude (a header whose east / west wording is wrong).</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(SiteLonDeg), nameof(SiteLonText))]
  private bool _flipSiteLon;

  /// <summary>The longitude "Use observatory site" applies, east positive.</summary>
  public double? SiteLonDeg => FactsData.SiteLonDeg is { } lon ? (FlipSiteLon ? -lon : lon) : null;

  /// <summary>"lon -17.8792° E · raw -17.8792 (sign from the header's LST)".</summary>
  public string SiteLonText
  {
    get
    {
      if (SiteLonDeg is not { } lon)
        return string.Empty;
      string why = FactsData.SiteLonFromLst == true ? "sign from the header's LST"
        : FactsData.SiteLonWestConvention == true ? "negated: the card says west"
        : "as written";
      string raw = FactsData.SiteLonRawDeg is { } r ? string.Format(Ci, "raw {0:0.####}", r) : "raw unknown";
      return string.Format(Ci, "lon {0:0.####}° E · {1} ({2}){3}", lon, raw, why, FlipSiteLon ? " · flipped" : "");
    }
  }

  // ── stretch ──

  public IReadOnlyList<FitsStretchMode> StretchModes { get; } =
    [FitsStretchMode.ZScale, FitsStretchMode.Percentile, FitsStretchMode.Manual];

  public IReadOnlyList<FitsTransfer> Transfers { get; } = [FitsTransfer.Linear, FitsTransfer.Log, FitsTransfer.Asinh];

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(HasStretchParams), nameof(Param0Label), nameof(Param1Label))]
  [NotifyCanExecuteChangedFor(nameof(ApplyStretchCommand))]
  private FitsStretchMode _stretchMode;

  [ObservableProperty]
  private FitsTransfer _transfer;

  /// <summary>Percentile: low %; manual: black (data units). Dot decimals.</summary>
  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(ApplyStretchCommand))]
  private string _param0Text = "0.5";

  /// <summary>Percentile: high %; manual: white.</summary>
  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(ApplyStretchCommand))]
  private string _param1Text = "99.5";

  public bool HasStretchParams => StretchMode != FitsStretchMode.ZScale;

  public string Param0Label => StretchMode == FitsStretchMode.Manual ? "Black" : "Low %";

  public string Param1Label => StretchMode == FitsStretchMode.Manual ? "White" : "High %";

  /// <summary>"black -1004.76 · white 1020.78" (the points the last stretch used).</summary>
  [ObservableProperty]
  private string _stretchResultText = string.Empty;

  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(ApplyStretchCommand))]
  private bool _isApplyingStretch;

  /// <summary>Outcome of the last purpose button (empty when none).</summary>
  [ObservableProperty]
  private string _statusMessage = string.Empty;

  private double? _lastBlack, _lastWhite;

  public FitsImageViewModel(DesktopBillboardItemViewModel row, FitsMetadata metadata, ulong assetId)
  {
    _row = row;
    Metadata = metadata;
    AssetId = assetId;
    TextureKey = row.TextureKey;
    if (metadata.StretchPoints is { } pts)
      ShowPoints(pts);
    // the billboard's saved stretch wins over what the texture renders right now
    var s = row.Current.FitsStretch ?? metadata.Stretch;
    _settingStretch = true;
    try
    {
      StretchMode = s.Mode;
      Transfer = s.Transfer;
      if (s.Mode == FitsStretchMode.Percentile && !(s.P0 < s.P1))
      {
        // the engine echoes no percentiles (a new billboard on a re-stretched texture): the defaults
        Param0Text = "0.5";
        Param1Text = "99.5";
      }
      else if (s.Mode != FitsStretchMode.ZScale)
      {
        Param0Text = Num(s.P0);
        Param1Text = Num(s.P1);
      }
    }
    finally
    {
      _settingStretch = false;
    }
    Facts = BuildFacts();
  }

  /// <summary>The Match camera button is the row's own.</summary>
  public IRelayCommand MatchCameraToImageCommand => _row.MatchCameraToImageCommand;

  private IBillboardViewportProbe? Viewport => _row.Owner?.Viewport;

  /// <summary>Re-evaluates the purpose buttons (the viewport changed).</summary>
  public void RefreshPilot()
  {
    SeekToObservationCommand.NotifyCanExecuteChanged();
    UseObservatorySiteCommand.NotifyCanExecuteChanged();
    OrientToImageCommand.NotifyCanExecuteChanged();
    SetUpComparisonCommand.NotifyCanExecuteChanged();
  }

  partial void OnFlipSiteLonChanged(bool value) => Facts = BuildFacts();

  partial void OnStretchModeChanged(FitsStretchMode value)
  {
    if (_settingStretch)
      return;
    // sensible starting parameters for the new mode
    if (value == FitsStretchMode.Percentile)
    {
      Param0Text = "0.5";
      Param1Text = "99.5";
    }
    else if (value == FitsStretchMode.Manual && _lastBlack is { } b && _lastWhite is { } w)
    {
      Param0Text = Num(b);
      Param1Text = Num(w);
    }
  }

  /// <summary>The stretch the controls describe; null while a parameter does not parse.</summary>
  public FitsStretch? RequestedStretch()
  {
    if (StretchMode == FitsStretchMode.ZScale)
      return new FitsStretch(FitsStretchMode.ZScale, Transfer, 0, 0);
    if (!TryParse(Param0Text, out double p0) || !TryParse(Param1Text, out double p1))
      return null;
    if (StretchMode == FitsStretchMode.Percentile && !(p0 >= 0 && p0 <= 100 && p1 >= 0 && p1 <= 100 && p0 < p1))
      return null;
    return new FitsStretch(StretchMode, Transfer, p0, p1);
  }

  private bool CanApplyStretch() => !IsApplyingStretch && _row.Owner?.Fits is not null && RequestedStretch() is not null;

  /// <summary>Re-stretches the texture off the UI thread; the billboard windows then re-copy it and
  /// every billboard of this texture records the stretch (saved, re-applied after a load).</summary>
  [RelayCommand(CanExecute = nameof(CanApplyStretch))]
  private async Task ApplyStretchAsync()
  {
    if (_row.Owner?.Fits is not { } fits || RequestedStretch() is not { } stretch)
      return;
    IsApplyingStretch = true;
    try
    {
      var result = await fits.ApplyAsync(AssetId, stretch, TextureKey);
      if (result is null)
        StretchResultText = "Stretch refused (library busy or bad parameters)";
      else
        ShowPoints(result);
    }
    finally
    {
      IsApplyingStretch = false;
    }
  }

  private void ShowPoints(FitsStretchResult pts)
  {
    _lastBlack = pts.Black;
    _lastWhite = pts.White;
    StretchResultText = string.Format(Ci, "black {0:G6} · white {1:G6}", pts.Black, pts.White);
  }

  partial void OnTransferChanged(FitsTransfer value) => ApplyStretchCommand.NotifyCanExecuteChanged();

  // ── purpose buttons ──

  private bool CanSeekToObservation() =>
    _row.Owner?.CanSeek == true && FactsData is { DateMidTdbCenturies: not null, DateMidTdbNanoseconds: not null };

  /// <summary>Seeks the timeline to the middle of the exposure (TDB).</summary>
  [RelayCommand(CanExecute = nameof(CanSeekToObservation))]
  private async Task SeekToObservationAsync()
  {
    if (!await SeekAsync())
      StatusMessage = "Seek refused (outside the loaded ephemeris?)";
  }

  private async Task<bool> SeekAsync() =>
    _row.Owner is { } o && FactsData is { DateMidTdbCenturies: { } c, DateMidTdbNanoseconds: { } ns }
    && await o.SeekTdbAsync(c, ns);

  private bool CanUseObservatorySite() => Viewport is not null && FactsData.SiteLatDeg is not null && SiteLonDeg is not null;

  /// <summary>The observer stands on the header's site (topocentric, latitude, longitude and the
  /// elevation when known).</summary>
  [RelayCommand(CanExecute = nameof(CanUseObservatorySite))]
  private void UseObservatorySite() => ApplySite();

  private bool ApplySite()
  {
    if (Viewport is not { } v || FactsData.SiteLatDeg is not { } lat || SiteLonDeg is not { } lon)
      return false;
    v.SetObserverSite(lat, lon, FactsData.SiteElevM);
    return true;
  }

  private bool CanOrientToImage() => Viewport is { IsEarthObserver: true } && FactsData.SkyCenter is not null;

  /// <summary>
  /// Aims the Earth observer at the image centre (a free look, fixed among the stars), turns the
  /// billboard like the sky in the view (<see cref="SkyMath.BillboardRotationDeg"/>) and moves it
  /// onto the viewport centre, where the image centre now is.
  /// </summary>
  [RelayCommand(CanExecute = nameof(CanOrientToImage))]
  private void OrientToImage()
  {
    if (!Orient())
      StatusMessage = "Orient refused: enter the Earth observer view";
  }

  private bool Orient()
  {
    if (Viewport is not { } v || FactsData.SkyCenter is not { } c)
      return false;
    if (!v.LockToDirection(SkyMath.EclipticDirection(c.RaDeg, c.DecDeg)))
      return false;
    double? rotation = FactsData.RotationDeg is { } pa
      ? SkyMath.BillboardRotationDeg(c.RaDeg, c.DecDeg, pa, v.ViewUpIsEclipticSouth, v.ViewUpIsCelestialNorth)
      : null;
    var centre = v.ViewportCenterDesktopPx;
    _row.Service.Update(_row.Id, b => b with
    {
      RotationDeg = rotation ?? b.RotationDeg,
      CenterX = centre?.X ?? b.CenterX,
      CenterY = centre?.Y ?? b.CenterY,
    });
    StatusMessage = FactsData.Mirrored == true
      ? "The image is mirrored (east right): a rotation cannot match it"
      : string.Empty;
    return true;
  }

  private bool CanSetUpComparison() => Viewport is not null;

  /// <summary>
  /// Everything in order: Earth observer → seek → site → match the field (the image's height fills
  /// the viewport's, aimed at the comet; the billboard fills it, bound) → orient (a free look at the
  /// WCS centre, the sky turned like the image) → centre (on the viewport centre once oriented, else
  /// on the comet) → lock the viewport for comparison. Steps without header facts are skipped and
  /// listed in <see cref="StatusMessage"/>.
  /// </summary>
  [RelayCommand(CanExecute = nameof(CanSetUpComparison))]
  private async Task SetUpComparisonAsync()
  {
    if (Viewport is not { } v || _row.Owner is not { } owner)
      return;
    if (!v.EnterEarthObserver())
    {
      StatusMessage = "Could not enter the Earth observer view";
      return;
    }
    var skipped = new List<string>();
    if (!CanSeekToObservation() || !await SeekAsync())
      skipped.Add("seek");
    if (!ApplySite())
      skipped.Add("site");
    owner.Probe();
    // match aims at the comet (tracking), so orient comes after it: the free look at the WCS centre
    // keeps the matched field
    if (!owner.MatchCameraToImage(_row.Id))
      skipped.Add("match");
    bool oriented = Orient();
    if (!oriented)
      skipped.Add("orient");
    // match centres on the comet; oriented, the image centre is the viewport centre
    if (oriented && v.ViewportCenterDesktopPx is { } centre)
      _row.Service.Update(_row.Id, b => b with { CenterX = centre.X, CenterY = centre.Y });
    else if (!owner.CenterOnComet(_row.Id))
      skipped.Add("center");
    v.IsComparisonLocked = true;
    owner.Probe();
    var mirrored = FactsData.Mirrored == true ? " The image is mirrored (east right)." : "";
    StatusMessage = (skipped.Count == 0
      ? "Comparison set up; the viewport is locked."
      : $"Comparison set up without: {string.Join(", ", skipped)}; the viewport is locked.") + mirrored;
  }

  // ── facts ──

  private IReadOnlyList<FitsFactItem> BuildFacts()
  {
    var f = FactsData;
    var list = new List<FitsFactItem>();
    void Add(string label, string? value)
    {
      if (!string.IsNullOrEmpty(value))
        list.Add(new FitsFactItem(label, value!));
    }

    Add("Object", f.Object);
    Add(f.DateSource ?? "DATE-OBS", f.DateObs);
    if (f.DateMidUtc is { } mid)
      Add("Mid-exposure", mid + (f.DateIsMid == true ? " (the card is mid-exposure)" : "")
        + (f.TimesysAssumed == true ? " · no TIMESYS: read as UTC" : ""));
    Add("Telescope", Join(" / ", f.Telescope, f.Instrument));
    Add("Filter", f.Filter);
    Add("Exposure", f.ExposureS is { } e ? string.Format(Ci, "{0:0.###} s", e) : null);
    Add("Plate scale", f.PixelScaleArcsec is { } s
      ? string.Format(Ci, "{0:0.#####}″/px", s) + (f.PixelScaleSource is { } src ? $" ({src})" : "")
      : null);
    Add("Field", f.FovWidthArcmin is { } fw && f.FovHeightArcmin is { } fh
      ? string.Format(Ci, "{0:F2}′ × {1:F2}′", fw, fh)
      : null);
    if (f.SiteLatDeg is { } lat)
      Add("Site", string.Format(Ci, "lat {0:0.####}°", lat)
        + (SiteLonDeg is not null ? " · " + SiteLonText : "")
        + (f.SiteElevM is { } el ? string.Format(Ci, " · {0:0.#} m", el) : "")
        + (f.SiteSource is { } ss ? $" ({ss})" : ""));
    if (f.CenterRaDeg is { } ra && f.CenterDecDeg is { } dec)
      Add("WCS centre", FormatRaDec(ra, dec) + (f.Radesys is { } rs ? $" {rs}" : ""));
    else if (f.SkyCenter is { } sc)
      Add("Pointing", FormatRaDec(sc.RaDeg, sc.DecDeg));
    Add("Rotation", f.RotationDeg is { } rot
      ? string.Format(Ci, "{0:0.###}° E of N (image up)", rot) + (f.Mirrored == true ? " · mirrored" : "")
      : null);
    Add("WCS", f.WcsError);
    return list;
  }

  /// <summary>"RA 09:02:15.22 (135.56340°) · Dec +28:09:24.8 (28.15690°)".</summary>
  internal static string FormatRaDec(double raDeg, double decDeg)
  {
    double h = ((raDeg % 360 + 360) % 360) / 15.0;
    int hh = (int)h;
    int mm = (int)((h - hh) * 60);
    double ss = ((h - hh) * 60 - mm) * 60;
    double ad = Math.Abs(decDeg);
    int dd = (int)ad;
    int dm = (int)((ad - dd) * 60);
    double ds = ((ad - dd) * 60 - dm) * 60;
    return string.Format(Ci, "RA {0:00}:{1:00}:{2:00.00} ({3:F5}°) · Dec {4}{5:00}:{6:00}:{7:00.0} ({8:F5}°)",
      hh, mm, ss, raDeg, decDeg < 0 ? "-" : "+", dd, dm, ds, decDeg);
  }

  private static string? Join(string sep, params string?[] parts)
  {
    var present = parts.Where(p => !string.IsNullOrEmpty(p)).ToArray();
    return present.Length == 0 ? null : string.Join(sep, present);
  }

  private static string Num(double v) => v.ToString("G9", Ci);

  private static bool TryParse(string text, out double v) =>
    double.TryParse(text.Trim(), NumberStyles.Float, Ci, out v) && !double.IsNaN(v) && !double.IsInfinity(v);
}
