using System;
using System.Collections.Generic;
using System.IO;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Attributes;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

/// <summary>Sky image maths shared with the native side (<c>scene/sky_image.rs</c>).</summary>
public static class SkyImageMath
{
  /// <summary>
  /// J2000 equator → ecliptic J2000 angle (ANISE / SPICE <c>ECLIPJ2000</c>, IAU 1976 84381.448″),
  /// the rotation the engine's world axes use.
  /// </summary>
  public const double ObliquityRad = 0.40909280422232897;

  /// <summary>Nanoseconds per Julian century (the <c>hifitime</c> <c>Duration</c> parts).</summary>
  public const long NanosecondsPerCentury = 36525L * 86400L * 1_000_000_000L;

  /// <summary>World (ecliptic J2000) unit direction of an ICRS right ascension / declination.</summary>
  public static Vector3d EclipticDirection(double raDeg, double decDeg)
  {
    double ra = raDeg * Math.PI / 180.0, dec = decDeg * Math.PI / 180.0;
    double x = Math.Cos(dec) * Math.Cos(ra), y = Math.Cos(dec) * Math.Sin(ra), z = Math.Sin(dec);
    double s = Math.Sin(ObliquityRad), c = Math.Cos(ObliquityRad);
    return new Vector3d(x, c * y + s * z, -s * y + c * z);
  }

  /// <summary>
  /// TDB microseconds since J2000 TDB → the <c>(centuries, nanoseconds)</c> parts
  /// <see cref="INativeRuntimeService.SeekEpochSync"/> takes (nanoseconds in [0, one century)).
  /// </summary>
  public static (short Centuries, ulong Nanoseconds) TdbMicrosToParts(long tdbUs)
  {
    long ns = checked(tdbUs * 1000L);
    long centuries = Math.DivRem(ns, NanosecondsPerCentury, out long rem);
    if (rem < 0)
    {
      centuries--;
      rem += NanosecondsPerCentury;
    }
    return ((short)centuries, (ulong)rem);
  }

  /// <summary>Middle of the exposure: DATE-OBS + EXPTIME / 2 (EXPTIME absent: DATE-OBS).</summary>
  public static long MidExposureTdbUs(long dateObsTdbUs, double? exptimeS) =>
    dateObsTdbUs + (long)Math.Round((exptimeS ?? 0.0) * 0.5e6);
}

/// <summary>
/// Telescope image overlay of the Earth-observer panel (WP5): loads a FITS (or PNG / JPEG with a
/// <c>.wcs</c> sidecar) registered on the sky by its WCS and drawn natively above the scene, below
/// the UI. Offers, from the header: seek to DATE-OBS + EXPTIME/2, use the observing site, and
/// "lock to image" (the telescope lock-in aimed at CRVAL). An offer whose header fields are missing
/// is disabled, its tooltip and <see cref="MissingHeaderMessage"/> say what is missing.
/// </summary>
[GenerateLocalizedStrings(keyPrefix: "Panels_SkyImage_", designTitle: "Sky image", designIcon: "✦")]
public partial class SkyImageOverlayViewModel : ObservableObject, ISkyImageOverlayViewModel, IDisposable
{
  /// <summary>File picker extensions.</summary>
  public static readonly string[] Extensions = ["fits", "fit", "fts", "png", "jpg", "jpeg"];

  /// <summary>Default asinh stretch of a FITS image (raw data); PNG / JPEG are already stretched.</summary>
  public const double DefaultFitsStretch = 10.0;

  private readonly INativeRuntimeService _runtimeService;
  private readonly IFileDialogService _fileDialogService;
  private readonly ITranslationService _translationService;
  private readonly Action<double, double, double?> _applySite;
  private readonly Func<Vector3d, bool> _lockToDirection;
  private readonly CompositeDisposable _disposables = [];

  /// <summary>Suppresses the parameter push while the VM sets its own properties.</summary>
  private bool _suppressPush;

  /// <summary>Facts of the loaded image (<c>null</c>: none).</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(
    nameof(HasImage),
    nameof(CanSeekToDateObs),
    nameof(CanUseSite),
    nameof(CanLockToImage),
    nameof(SeekToolTip),
    nameof(SiteToolTip),
    nameof(LockToolTip),
    nameof(MissingHeaderMessage),
    nameof(HasMissingHeader))]
  [NotifyCanExecuteChangedFor(
    nameof(SeekToDateObsCommand),
    nameof(UseSiteFromHeaderCommand),
    nameof(LockToImageCommand),
    nameof(RemoveImageCommand))]
  private SkyImageInfo? _info;

  [ObservableProperty]
  private string _imageName = string.Empty;

  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(LoadImageCommand))]
  private bool _isLoading;

  /// <summary>Outcome of the last action (empty when none).</summary>
  [ObservableProperty]
  private string _statusMessage = string.Empty;

  /// <summary>[0, 1].</summary>
  [ObservableProperty]
  private double _opacity = 0.7;

  /// <summary>asinh softening (0 = linear).</summary>
  [ObservableProperty]
  private double _stretch = DefaultFitsStretch;

  [ObservableProperty]
  private bool _isOverlayVisible = true;

  /// <summary>An image is loaded (drawn when it has a WCS).</summary>
  public bool HasImage => Info is not null;

  public bool CanSeekToDateObs => Info?.DateObsTdbUs is not null;

  public bool CanUseSite => Info is { SiteLatDeg: not null, SiteLonDeg: not null };

  public bool CanLockToImage => Info?.HasWcs == true;

  public string SeekToolTip => CanSeekToDateObs ? StrSeekTip : StrMissingDate;

  public string SiteToolTip => CanUseSite ? StrSiteTip : StrMissingSite;

  public string LockToolTip => CanLockToImage ? StrLockTip : StrMissingWcs;

  /// <summary>
  /// Breadcrumb after a load: the header facts the offers need and the image lacks (empty when
  /// nothing is missing or no image).
  /// </summary>
  public string MissingHeaderMessage
  {
    get
    {
      if (Info is null) return string.Empty;
      var missing = new List<string>();
      if (!Info.HasWcs) missing.Add(StrMissingWcs);
      if (!CanSeekToDateObs) missing.Add(StrMissingDate);
      if (!CanUseSite) missing.Add(StrMissingSite);
      return string.Join("\n", missing);
    }
  }

  public bool HasMissingHeader => MissingHeaderMessage.Length > 0;

  /// <param name="applySite">sets the observer's latitude / longitude (degrees, east positive) and,
  /// when the header has it, the elevation (metres)</param>
  /// <param name="lockToDirection">telescope lock-in at a world (ecliptic J2000) direction;
  /// <c>false</c> when refused (not in the Earth observer view)</param>
  public SkyImageOverlayViewModel(
    INativeRuntimeService runtimeService,
    IFileDialogService fileDialogService,
    ITranslationService translationService,
    ISchedulerProvider schedulerProvider,
    Action<double, double, double?> applySite,
    Func<Vector3d, bool> lockToDirection)
  {
    _runtimeService = runtimeService;
    _fileDialogService = fileDialogService;
    _translationService = translationService;
    _applySite = applySite;
    _lockToDirection = lockToDirection;
    RefreshStrings();
    _translationService.CultureChanged
      .Skip(1)
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(_ =>
      {
        RefreshStrings();
        OnPropertyChanged(nameof(SeekToolTip));
        OnPropertyChanged(nameof(SiteToolTip));
        OnPropertyChanged(nameof(LockToolTip));
        OnPropertyChanged(nameof(MissingHeaderMessage));
      })
      .AddDisposableTo(_disposables);
  }

  private bool CanLoad() => !IsLoading;

  /// <summary>"Overlay image…": picks a file and loads it.</summary>
  [RelayCommand(CanExecute = nameof(CanLoad))]
  private async Task LoadImageAsync()
  {
    var path = await _fileDialogService.ShowOpenFileDialogAsync(StrDialogTitle, Extensions);
    if (string.IsNullOrEmpty(path)) return;
    await LoadPathAsync(path!);
  }

  /// <summary>Loads <paramref name="path"/> off the UI thread and applies the display parameters.</summary>
  public async Task LoadPathAsync(string path, string? sidecarPath = null)
  {
    IsLoading = true;
    try
    {
      var info = await Task.Run(() => _runtimeService.LoadSkyImage(path, sidecarPath));
      ImageName = Path.GetFileName(path);
      if (info is null)
      {
        StatusMessage = $"{StrStatusFailed}: {ImageName}";
        return;
      }
      Info = info;
      _suppressPush = true;
      try
      {
        // PNG / JPEG are display-stretched already: linear
        Stretch = info.IsRaster ? 0.0 : DefaultFitsStretch;
      }
      finally
      {
        _suppressPush = false;
      }
      PushParams();
      StatusMessage = info.SipIgnored
        ? $"{StrStatusLoaded}: {ImageName}. {StrSipIgnored}"
        : $"{StrStatusLoaded}: {ImageName}";
    }
    finally
    {
      IsLoading = false;
    }
  }

  [RelayCommand(CanExecute = nameof(CanSeekToDateObs))]
  private void SeekToDateObs()
  {
    if (Info?.DateObsTdbUs is not { } t0) return;
    var (c, ns) = SkyImageMath.TdbMicrosToParts(SkyImageMath.MidExposureTdbUs(t0, Info.ExptimeS));
    if (!_runtimeService.SeekEpochSync(c, ns))
      StatusMessage = StrSeekFailed;
  }

  [RelayCommand(CanExecute = nameof(CanUseSite))]
  private void UseSiteFromHeader()
  {
    if (Info is not { SiteLatDeg: { } lat, SiteLonDeg: { } lon }) return;
    _applySite(lat, lon, Info.SiteElevM);
  }

  [RelayCommand(CanExecute = nameof(CanLockToImage))]
  private void LockToImage()
  {
    if (Info is not { HasWcs: true } info) return;
    if (!_lockToDirection(SkyImageMath.EclipticDirection(info.CrvalRaDeg, info.CrvalDecDeg)))
      StatusMessage = StrLockFailed;
  }

  [RelayCommand(CanExecute = nameof(HasImage))]
  private void RemoveImage()
  {
    _runtimeService.RemoveSkyImage();
    Info = null;
    ImageName = string.Empty;
    StatusMessage = string.Empty;
  }

  partial void OnOpacityChanged(double value) => PushParams();

  partial void OnStretchChanged(double value) => PushParams();

  partial void OnIsOverlayVisibleChanged(bool value) => PushParams();

  private void PushParams()
  {
    if (_suppressPush || Info is not { HasWcs: true }) return;
    _runtimeService.SetSkyImageParams(
      (float)Math.Min(Math.Max(Opacity, 0.0), 1.0),
      (float)Math.Max(Stretch, 0.0),
      IsOverlayVisible);
  }

  public void Dispose() => _disposables.Dispose();
}
