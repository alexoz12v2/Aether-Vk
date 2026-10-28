using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Linq;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

/// <summary>The displays of the desktop (implemented by the app from Avalonia's <c>Screens</c>).</summary>
public interface IScreenInfoProvider
{
  /// <summary>Every display, desktop physical pixels.</summary>
  IReadOnlyList<ScreenInfo> Screens { get; }

  /// <summary>The display showing the main window (new billboards open there).</summary>
  ScreenInfo MainWindowScreen { get; }
}

/// <summary>
/// What a billboard needs from the Earth observer viewport to match a telescope image: the
/// viewport's scale, where the comet is on the desktop, and a way to set the scale.
/// </summary>
public interface IBillboardViewportProbe
{
  /// <summary>Viewport arcsec per physical px in the Earth observer; null elsewhere.</summary>
  double? ArcsecPerPx { get; }

  /// <summary>The observed comet on the desktop (physical px) while it is inside the viewport.</summary>
  PointD? CometDesktopPx { get; }

  /// <summary>Sets the viewport's arcsec per px (the projection); false when it cannot.</summary>
  bool MatchArcsecPerPx(double arcsecPerPx);

  // ── FITS purpose buttons ──

  /// <summary>The viewport is in the Earth observer (<see cref="CameraMode.EarthPosition"/>).</summary>
  bool IsEarthObserver { get; }

  /// <summary>Switches the viewport to the Earth observer; true once it is in it.</summary>
  bool EnterEarthObserver();

  /// <summary>Puts the observer on a surface site (topocentric on, then elevation and latitude /
  /// longitude, degrees, east positive).</summary>
  void SetObserverSite(double latDeg, double lonDeg, double? elevM);

  /// <summary>Free look along a world (ecliptic J2000) direction; false outside the Earth observer.</summary>
  bool LockToDirection(Vector3d worldDirection);

  /// <summary>The view's up leans to the ecliptic south pole (a topocentric site south of the
  /// equator), else the north one.</summary>
  bool ViewUpIsEclipticSouth { get; }

  /// <summary>The viewport's centre on the desktop (physical px); null until it is known.</summary>
  PointD? ViewportCenterDesktopPx { get; }

  /// <summary>Mouse orbit / pan / zoom refused so the view cannot drift from a matched image.</summary>
  bool IsComparisonLocked { get; set; }
}

/// <summary><see cref="IBillboardViewportProbe"/> over a viewport's <see cref="CameraService"/>.</summary>
public sealed class CameraBillboardProbe(CameraService camera) : IBillboardViewportProbe
{
  public double? ArcsecPerPx => camera.GetEarthObserverScale()?.ArcsecPerPx;

  public PointD? CometDesktopPx =>
    camera.ViewportScreenOriginPx is { } o && camera.GetCometScreenPosition() is { OnScreen: true } c
      ? new PointD(o.X + c.XPx, o.Y + c.YPx)
      : null;

  public bool MatchArcsecPerPx(double arcsecPerPx) => camera.SetEarthObserverArcsecPerPx(arcsecPerPx);

  public bool IsEarthObserver => camera.CurrentMode == CameraMode.EarthPosition;

  public bool EnterEarthObserver()
  {
    if (!IsEarthObserver)
      camera.SetCameraMode(CameraMode.EarthPosition);
    return IsEarthObserver;
  }

  public void SetObserverSite(double latDeg, double lonDeg, double? elevM)
  {
    if (elevM is { } elev)
      camera.SetEarthObserverElevation(elev);
    camera.SetEarthObserverTopocentric(true);
    camera.SetEarthObserverLatLon((float)latDeg, (float)lonDeg); // the Settings site fields follow
  }

  public bool LockToDirection(Vector3d worldDirection) => camera.LockEarthObserverToDirection(worldDirection);

  public bool ViewUpIsEclipticSouth
  {
    get
    {
      // CameraService picks the pole from the site latitude while topocentric (UpLatDeg)
      float lat = 0f;
      using (camera.EarthObserverSiteChanged.Take(1).Subscribe(s => lat = s.LatDeg)) { }
      return camera.IsEarthObserverTopocentric && lat < 0f;
    }
  }

  public PointD? ViewportCenterDesktopPx =>
    camera.ViewportScreenOriginPx is { } o
      ? new PointD(o.X + camera.ViewportWidthPx / 2.0, o.Y + camera.ViewportHeightPx / 2.0)
      : null;

  public bool IsComparisonLocked
  {
    get => camera.IsComparisonLocked;
    set => camera.IsComparisonLocked = value;
  }
}

/// <summary>
/// "Billboards" section of the Settings tab: create a billboard from an imported texture, and one
/// editable row per billboard (desktop position, size, opacity, rotation, visibility). Edits go to
/// <see cref="DesktopBillboardService"/>; changes made by dragging the billboard windows come back
/// from it, so the rows and the windows always agree.
/// </summary>
public partial class DesktopBillboardsViewModel : ObservableObject, IDisposable
{
  private readonly DesktopBillboardService _service;
  private readonly IScreenInfoProvider _screens;
  private readonly CompositeDisposable _disposables = [];
  private readonly ISchedulerProvider? _schedulerProvider;
  private ImmutableArray<ImportedAsset> _textures = ImmutableArray<ImportedAsset>.Empty;
  private IBillboardViewportProbe? _probe;
  private IDisposable? _probeTimer;
  private readonly Func<short, ulong, Task<bool>>? _seekTdb;

  /// <summary>How often the viewport's scale and the comet's position are read.</summary>
  public static readonly TimeSpan ProbeInterval = TimeSpan.FromMilliseconds(250);

  /// <summary>The Earth observer viewport's arcsec per px (null outside it).</summary>
  public double? ViewportArcsecPerPx { get; private set; }

  /// <summary>The observed comet on the desktop while it is on screen.</summary>
  public PointD? CometDesktopPx { get; private set; }

  /// <summary>Textures a billboard can show (uncompressed: the window draws their texels).</summary>
  public ObservableCollection<ImportedAsset> TextureOptions { get; } = [];

  public ObservableCollection<DesktopBillboardItemViewModel> Items { get; } = [];

  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(CreateCommand))]
  private ImportedAsset? _selectedTexture;

  [ObservableProperty]
  private bool _hasTextures;

  public DesktopBillboardsViewModel(
    DesktopBillboardService service,
    AssetLibraryService assetLibrary,
    IScreenInfoProvider screens,
    ISchedulerProvider schedulerProvider,
    FitsImageService? fits = null,
    Func<short, ulong, Task<bool>>? seekTdb = null)
  {
    _service = service;
    _screens = screens;
    _schedulerProvider = schedulerProvider;
    Fits = fits;
    _seekTdb = seekTdb;
    assetLibrary.Textures
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(OnTextures)
      .AddDisposableTo(_disposables);
    service.Billboards
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(Sync)
      .AddDisposableTo(_disposables);
  }

  /// <summary>Design-time / derived-class constructor: a private service on one 1920×1080 display,
  /// rows kept in sync synchronously (the previewer and the headless view tests create rows).</summary>
  protected DesktopBillboardsViewModel()
  {
    _service = new DesktopBillboardService();
    _screens = new FixedScreens();
    _service.Billboards.Subscribe(Sync).AddDisposableTo(_disposables);
  }

  /// <summary>The billboards this view model edits.</summary>
  public DesktopBillboardService Service => _service;

  /// <summary>FITS metadata and stretches (null: no FITS rows, e.g. design time).</summary>
  public FitsImageService? Fits { get; }

  /// <summary>The viewport the rows pilot (null until one is attached).</summary>
  internal IBillboardViewportProbe? Viewport => _probe;

  /// <summary>A timeline seek to a TDB epoch is available ("Seek to observation").</summary>
  internal bool CanSeek => _seekTdb is not null;

  /// <summary>Seeks the timeline to a TDB epoch (the engine's centuries / nanoseconds parts).</summary>
  internal Task<bool> SeekTdbAsync(short centuries, ulong nanoseconds) =>
    _seekTdb is { } seek ? seek(centuries, nanoseconds) : Task.FromResult(false);

  /// <summary>
  /// Pilots the billboards from the viewport <paramref name="probe"/> (the app has one viewport):
  /// reads its scale and the comet every <see cref="ProbeInterval"/>, keeps bound billboards matched
  /// and enables the rows' Center / Apply / Match commands. Without <paramref name="poll"/> only
  /// <see cref="Probe"/> reads it (tests).
  /// </summary>
  public void AttachViewport(IBillboardViewportProbe probe, bool poll = true)
  {
    _probe = probe;
    _probeTimer?.Dispose();
    _probeTimer = null;
    if (poll && _schedulerProvider is { } s)
      _probeTimer = Observable.Interval(ProbeInterval, s.Background)
        .ObserveOn(s.MainThread)
        .Subscribe(_ => Probe());
    Probe();
  }

  /// <summary>Reads the viewport once (the timer's work).</summary>
  public void Probe()
  {
    if (_probe is not { } p)
      return;
    ViewportArcsecPerPx = p.ArcsecPerPx is { } a && a > 0 && !double.IsInfinity(a) ? a : null;
    CometDesktopPx = p.CometDesktopPx;
    if (ViewportArcsecPerPx is { } scale)
      _service.RescaleBound(scale);
    foreach (var item in Items)
      item.RefreshPilot();
  }

  internal bool CenterOnComet(Guid id) =>
    CometDesktopPx is { } c && _service.Update(id, b => b with { CenterX = c.X, CenterY = c.Y });

  internal bool ApplyPhysicalSize(Guid id) =>
    ViewportArcsecPerPx is { } scale && _service.ApplyPhysicalSize(id, scale);

  /// <summary>"Match camera to image": the viewport takes the image's plate scale, the billboard
  /// its native pixel size (bound) and the comet's position.</summary>
  internal bool MatchCameraToImage(Guid id)
  {
    if (_probe is not { } p || _service.Find(id)?.ImageArcsecPerPx is not { } scale || !p.MatchArcsecPerPx(scale))
      return false;
    _service.ApplyPhysicalSize(id, scale);
    Probe();
    CenterOnComet(id);
    return true;
  }

  private bool CanCreate() => SelectedTexture is not null;

  [RelayCommand(CanExecute = nameof(CanCreate))]
  private void Create()
  {
    if (SelectedTexture is { } t)
      _service.Create(t, _screens.MainWindowScreen);
  }

  private void OnTextures(ImmutableArray<ImportedAsset> textures)
  {
    _textures = textures;
    var displayable = textures.Where(t => t.IsDisplayable).ToList();
    var keep = SelectedTexture?.Key;
    TextureOptions.Clear();
    foreach (var t in displayable)
      TextureOptions.Add(t);
    SelectedTexture = displayable.FirstOrDefault(t => t.Key == keep) ?? displayable.FirstOrDefault();
    HasTextures = displayable.Count > 0;
    foreach (var item in Items)
      item.SetTexture(Lookup(item.TextureKey));
  }

  private ImportedAsset? Lookup(string key) => _textures.FirstOrDefault(t => t.Key == key);

  private void Sync(ImmutableArray<DesktopBillboard> billboards)
  {
    for (int i = Items.Count - 1; i >= 0; i--)
      if (!billboards.Any(b => b.Id == Items[i].Id))
      {
        Items[i].Dispose();
        Items.RemoveAt(i);
      }
    foreach (var b in billboards)
    {
      var item = Items.FirstOrDefault(x => x.Id == b.Id);
      if (item is null)
      {
        item = new DesktopBillboardItemViewModel(b, _service, _screens, this);
        Items.Add(item);
      }
      item.Apply(b);
      item.SetTexture(Lookup(b.TextureKey));
    }
  }

  public void Dispose()
  {
    _probeTimer?.Dispose();
    _disposables.Dispose();
    foreach (var item in Items)
      item.Dispose();
  }

  private sealed class FixedScreens : IScreenInfoProvider
  {
    private static readonly ScreenInfo Screen = new(new RectI(0, 0, 1920, 1080), 1, true);
    public IReadOnlyList<ScreenInfo> Screens { get; } = [Screen];
    public ScreenInfo MainWindowScreen => Screen;
  }
}

/// <summary>One billboard row of the Settings "Billboards" section.</summary>
public partial class DesktopBillboardItemViewModel : ObservableObject, IDisposable
{
  private readonly DesktopBillboardService _service;
  private readonly IScreenInfoProvider _screens;
  private readonly DesktopBillboardsViewModel? _owner;
  private bool _applying;
  private DesktopBillboard _current;

  public Guid Id { get; }

  /// <summary>Physical size: the image's plate scale in arcsec per image pixel, as typed (empty:
  /// not set; dot decimals).</summary>
  [ObservableProperty]
  private string _imageArcsecPerPxText = string.Empty;

  /// <summary>"6.07′ × 6.07′ · 1200 × 1200 px", or a hint while the scale is unset.</summary>
  [ObservableProperty]
  private string _physicalSizeLabel = string.Empty;

  /// <summary>The size follows the viewport's scale.</summary>
  [ObservableProperty]
  private bool _isPhysicalSizeBound;

  /// <summary>The "FITS image" part of the row (null: the texture is not FITS).</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(IsFits))]
  private FitsImageViewModel? _fits;

  public bool IsFits => Fits is not null;

  /// <summary>The billboard as the service last reported it.</summary>
  internal DesktopBillboard Current => _current;

  internal DesktopBillboardsViewModel? Owner => _owner;

  internal DesktopBillboardService Service => _service;

  public string TextureKey { get; private set; }

  [ObservableProperty]
  private string _textureLabel = string.Empty;

  [ObservableProperty]
  private AssetThumbnail? _thumbnail;

  /// <summary>The texture was unloaded (the window shows a placeholder until it is imported again).</summary>
  [ObservableProperty]
  private bool _isMissing;

  /// <summary>Centre, desktop px.</summary>
  [ObservableProperty]
  private double _x;

  [ObservableProperty]
  private double _y;

  [ObservableProperty]
  private double _width;

  [ObservableProperty]
  private double _height;

  /// <summary>Width and height change together (the image's aspect is kept).</summary>
  [ObservableProperty]
  private bool _isAspectLocked = true;

  [ObservableProperty]
  private double _alpha;

  [ObservableProperty]
  private double _rotation;

  [ObservableProperty]
  private bool _isVisible;

  /// <summary>"Display 2 · 640, 360" (the centre on its display), or "off screen".</summary>
  [ObservableProperty]
  private string _displayLabel = string.Empty;

  public DesktopBillboardItemViewModel(
    DesktopBillboard b, DesktopBillboardService service, IScreenInfoProvider screens,
    DesktopBillboardsViewModel? owner = null)
  {
    Id = b.Id;
    TextureKey = b.TextureKey;
    _service = service;
    _screens = screens;
    _owner = owner;
    _current = b;
    Apply(b);
  }

  /// <summary>Shows the service's state (no edit is sent back).</summary>
  public void Apply(DesktopBillboard b)
  {
    _applying = true;
    try
    {
      _current = b;
      TextureKey = b.TextureKey;
      X = b.CenterX;
      Y = b.CenterY;
      Width = b.WidthPx;
      Height = b.HeightPx;
      Alpha = b.Alpha;
      Rotation = b.RotationDeg;
      IsVisible = b.Visible;
      var (display, local) = DesktopBillboardGeometry.ToDisplayLocal(_screens.Screens, new PointD(b.CenterX, b.CenterY));
      DisplayLabel = display < 0 ? "off screen" : $"Display {display + 1} · {local.X:F0}, {local.Y:F0}";
      IsPhysicalSizeBound = b.PhysicalSizeBound;
      if (ParseScale(ImageArcsecPerPxText) != b.ImageArcsecPerPx)
        ImageArcsecPerPxText = b.ImageArcsecPerPx is { } a ? a.ToString("G6", CultureInfo.InvariantCulture) : string.Empty;
      PhysicalSizeLabel = FormatPhysicalSize(b);
    }
    finally
    {
      _applying = false;
    }
    RefreshPilot();
  }

  /// <summary>Re-evaluates the Center / Apply / Match commands (the viewport changed).</summary>
  public void RefreshPilot()
  {
    CenterOnCometCommand.NotifyCanExecuteChanged();
    ApplyPhysicalSizeCommand.NotifyCanExecuteChanged();
    MatchCameraToImageCommand.NotifyCanExecuteChanged();
    Fits?.RefreshPilot();
  }

  private static double? ParseScale(string text) =>
    double.TryParse(text.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double v) && v > 0 && !double.IsInfinity(v)
      ? v
      : null;

  internal static string FormatPhysicalSize(DesktopBillboard b)
  {
    if (b.ImageWidthPx <= 0 || b.ImageHeightPx <= 0)
      return "Image size unknown";
    if (b.ImageArcsecPerPx is not { } a)
      return $"{b.ImageWidthPx} × {b.ImageHeightPx} px · set the plate scale (″/px) to size it on the sky";
    var ci = CultureInfo.InvariantCulture;
    return string.Format(ci, "{0:F2}′ × {1:F2}′ · {2} × {3} px",
      b.ImageWidthPx * a / 60.0, b.ImageHeightPx * a / 60.0, b.ImageWidthPx, b.ImageHeightPx);
  }

  partial void OnImageArcsecPerPxTextChanged(string value)
  {
    if (_applying)
      return;
    var scale = ParseScale(value);
    if (scale is null && !string.IsNullOrWhiteSpace(value))
      return; // still typing: keep the last valid scale
    // a new scale changes the physical size: the binding re-applies with it
    Send(b => b with { ImageArcsecPerPx = scale, PhysicalSizeBound = b.PhysicalSizeBound && scale is not null });
  }

  private bool CanCenterOnComet() => _owner?.CometDesktopPx is not null;

  /// <summary>Moves the billboard's centre onto the comet as the viewport draws it.</summary>
  [RelayCommand(CanExecute = nameof(CanCenterOnComet))]
  private void CenterOnComet() => _owner?.CenterOnComet(Id);

  private bool CanApplyPhysicalSize() =>
    !_current.PhysicalSizeBound && _owner?.ViewportArcsecPerPx is { } s
    && DesktopBillboardGeometry.PhysicalSize(_current, s) is not null;

  /// <summary>Sizes the billboard to the sky its image covers and keeps it so (until a manual
  /// resize).</summary>
  [RelayCommand(CanExecute = nameof(CanApplyPhysicalSize))]
  private void ApplyPhysicalSize() => _owner?.ApplyPhysicalSize(Id);

  private bool CanMatchCameraToImage() =>
    _owner?.ViewportArcsecPerPx is not null && _current.ImageArcsecPerPx is not null
    && DesktopBillboardGeometry.PhysicalSize(_current, _current.ImageArcsecPerPx.Value) is not null;

  /// <summary>The viewport takes the image's plate scale; the billboard shows the image 1:1 on
  /// the comet.</summary>
  [RelayCommand(CanExecute = nameof(CanMatchCameraToImage))]
  private void MatchCameraToImage() => _owner?.MatchCameraToImage(Id);

  public void SetTexture(ImportedAsset? texture)
  {
    // a billboard from an older save learns its image's native size
    if (texture is { A: > 0, B: > 0 } && (_current.ImageWidthPx <= 0 || _current.ImageHeightPx <= 0))
      _service.Update(Id, b => b with
      {
        ImageWidthPx = (int)Math.Min(int.MaxValue, texture.A),
        ImageHeightPx = (int)Math.Min(int.MaxValue, texture.B),
      });
    IsMissing = texture is null;
    TextureLabel = texture?.Label ?? TextureKey;
    Thumbnail = texture?.Thumbnail;
    UpdateFits(texture);
  }

  /// <summary>Reads the FITS metadata once per asset id (a busy library keeps what is shown).</summary>
  private void UpdateFits(ImportedAsset? texture)
  {
    if (_owner?.Fits is not { } fits)
      return; // design time: ShowFits decides
    if (texture is null)
    {
      Fits = null;
      return;
    }
    if (Fits?.AssetId == texture.Id)
      return;
    var meta = fits.GetMetadata(texture.Id);
    if (meta is null)
    {
      if (Fits is not null && Fits.TextureKey != texture.Key)
        Fits = null;
      return;
    }
    ShowFits(meta, texture.Id);
  }

  /// <summary>
  /// Shows the FITS part for <paramref name="meta"/> (design time and view tests call it directly).
  /// The first time, a billboard without a plate scale takes the header's.
  /// </summary>
  public void ShowFits(FitsMetadata meta, ulong assetId)
  {
    Fits = new FitsImageViewModel(this, meta, assetId);
    if (!_scaleAutoFilled && _current.ImageArcsecPerPx is null && meta.Facts.PixelScaleArcsec is { } scale && scale > 0)
      _service.Update(Id, b => b with { ImageArcsecPerPx = scale });
    _scaleAutoFilled = true;
  }

  private bool _scaleAutoFilled;

  partial void OnXChanged(double value) => Send(b => b with { CenterX = value });

  partial void OnYChanged(double value) => Send(b => b with { CenterY = value });

  partial void OnWidthChanged(double oldValue, double newValue)
  {
    if (_applying)
      return;
    if (IsAspectLocked && oldValue > 0)
      Send(b => b with { WidthPx = newValue, HeightPx = b.HeightPx * newValue / oldValue });
    else
      Send(b => b with { WidthPx = newValue });
  }

  partial void OnHeightChanged(double oldValue, double newValue)
  {
    if (_applying)
      return;
    if (IsAspectLocked && oldValue > 0)
      Send(b => b with { HeightPx = newValue, WidthPx = b.WidthPx * newValue / oldValue });
    else
      Send(b => b with { HeightPx = newValue });
  }

  partial void OnAlphaChanged(double value) => Send(b => b with { Alpha = value });

  partial void OnRotationChanged(double value) => Send(b => b with { RotationDeg = value });

  partial void OnIsVisibleChanged(bool value) => Send(b => b with { Visible = value });

  [RelayCommand]
  private void Delete() => _service.Remove(Id);

  private void Send(Func<DesktopBillboard, DesktopBillboard> edit)
  {
    if (_applying)
      return;
    _service.Update(Id, edit);
  }

  public void Dispose()
  {
  }
}
