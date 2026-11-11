using System;
using System.Globalization;
using System.Reactive.Linq;
using AetherVk.Logic.Attributes;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// The compact panel's "Nucleus seen from Earth" view: a second, small live viewport
/// (<see cref="ViewportRole.NucleusMini"/>) that shows the nucleus as a telescope on Earth sees it
/// (celestial north up, lit by the Sun only), without dust and without labels.
/// <list type="bullet">
///   <item>Its native view is created once, the first time it is shown expanded in the compact
///     layout, and removed only when the application shuts down.</item>
///   <item>The engine creates it paused. This class sets the camera's flags and the view from
///     Earth first and only then resumes it: it never draws a frame of anything else.</item>
///   <item>It is paused again while collapsed, in the classic layout or while the window is
///     minimized, and resumed when shown.</item>
///   <item>Dragging orbits the camera about the nucleus; <see cref="ResetToEarthCommand"/> hands it
///     back to the view from Earth, which follows the clock.</item>
/// </list>
/// Runs on the UI thread (pointer events are dispatched to it).
/// </summary>
[GenerateLocalizedStrings(keyPrefix: "NucleusMini_", designTitle: "Nucleus seen from Earth")]
public partial class NucleusMiniViewModel : ObservableObject, IDisposable
{
  /// <summary>The largest side of the square view, DIPs.</summary>
  public const double MaxSide = 320.0;

  /// <summary>The panel's gutters left free beside the view, DIPs (both sides together).</summary>
  public const double PanelGutter = 16.0;

  /// <summary>Vertical field of view of the mini camera (a narrow one: little perspective).</summary>
  internal const float FovRad = (float)(20.0 * Math.PI / 180.0);

  /// <summary>Radians the camera orbits per dragged pixel.</summary>
  internal const double OrbitRadPerPixel = 0.01;

  /// <summary>The view's frame-rate cap, µs between two frames (5 fps): it shows a slowly
  /// turning nucleus, and every frame it draws is time the main view waits for on the shared
  /// render thread (at 10 fps the main view lost 11 % of its frame rate on a GTX 1070).</summary>
  internal const ulong FrameIntervalUs = 200_000;
  private const double AuToKm = 149_597_870.7;
  private const double DefaultRadiusKm = 5.0;

  private readonly ITranslationService _translationService;
  private readonly INativeRuntimeService _runtime;
  private readonly PreferencesService? _preferences;
  private readonly IUiThreadDispatcher? _dispatcher;
  private readonly Func<Action<double, double, bool, bool>, VulkanViewportControlViewModel> _viewportFactory;
  private readonly IDisposable? _cultureSubscription;
  private readonly IDisposable? _positionSubscription;

  // the camera's flags are set (once per attached engine)
  private bool _flagsSet;
  // what the engine was last told (it starts paused)
  private bool _active;
  private bool _disposed;
  private double _radiusKm;
  private bool _cometCommitted;

  // free view: the camera's place relative to the nucleus (unit vector, world axes) and its up
  private Vector3d _orbitDir;
  private Vector3d _orbitUp;
  private bool _dragging;
  private double _lastX, _lastY;

  /// <summary>The side of the square view for a panel <paramref name="panelWidth"/> wide:
  /// min(width − <see cref="PanelGutter"/>, <see cref="MaxSide"/>), never negative.</summary>
  public static double HostSide(double panelWidth) =>
    double.IsNaN(panelWidth) ? 0 : Math.Max(0.0, Math.Min(panelWidth - PanelGutter, MaxSide));

  /// <param name="runtime">the engine</param>
  /// <param name="viewportFactory">builds the native view's model in the
  /// <see cref="ViewportRole.NucleusMini"/> role, with the given pointer callback</param>
  /// <param name="translation">UI strings (null: English)</param>
  /// <param name="preferences">remembers <see cref="IsExpanded"/> (null: not remembered)</param>
  /// <param name="dispatcher">the UI thread (null: calls run where they arrive)</param>
  /// <param name="cometPosition">the comet's place, AU from the Sun (null: 1 AU is assumed); the
  /// exposure takes out its sunlight's (1 AU / r)²</param>
  public NucleusMiniViewModel(
    INativeRuntimeService runtime,
    Func<Action<double, double, bool, bool>, VulkanViewportControlViewModel> viewportFactory,
    ITranslationService? translation = null,
    PreferencesService? preferences = null,
    IUiThreadDispatcher? dispatcher = null,
    IObservable<System.Numerics.Vector3?>? cometPosition = null)
  {
    _runtime = runtime;
    _viewportFactory = viewportFactory;
    _preferences = preferences;
    _dispatcher = dispatcher;
    _translationService = translation ?? new TranslationService(
      new System.Resources.ResourceManager("AetherVk.Logic.Resources.AppStrings", typeof(ITranslationService).Assembly),
      new CultureInfo("en"));
    _isExpanded = preferences?.NucleusMiniExpanded ?? true;
    _positionSubscription = cometPosition?.Subscribe(p =>
    {
      if (p is { } v && v.Length() > 1e-6f)
        OnUi(() => SetSunDistanceAu(v.Length()));
    });
    RefreshStrings();
    if (translation is not null)
    {
      _cultureSubscription = translation.CultureChanged
        .Skip(1)
        .Subscribe(_ => OnUi(() =>
        {
          RefreshStrings();
          OnPropertyChanged(nameof(CaptionText));
        }));
    }
  }

  // ── What the view shows ────────────────────────────────────────────────────

  /// <summary>The expander is open (remembered in the preferences, not in a save).</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(IsShown))]
  private bool _isExpanded;

  partial void OnIsExpandedChanged(bool value)
  {
    _preferences?.SetNucleusMiniExpanded(value);
    Refresh();
  }

  /// <summary>The compact layout is the one shown (the panel is hidden in the classic one).</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(IsShown))]
  private bool _isLayoutCompact;

  partial void OnIsLayoutCompactChanged(bool value) => Refresh();

  /// <summary>The main window is minimized.</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(IsShown))]
  private bool _isWindowMinimized;

  partial void OnIsWindowMinimizedChanged(bool value) => Refresh();

  /// <summary>The view is on screen: expanded, in the compact layout, the window not minimized.</summary>
  public bool IsShown => IsExpanded && IsLayoutCompact && !IsWindowMinimized;

  /// <summary>The native view's model; null until the view is first shown.</summary>
  [ObservableProperty]
  private VulkanViewportControlViewModel? _viewport;

  /// <summary>The engine poses the camera on the Earth → comet line, or the user orbits it: there
  /// is a nucleus to look at. False without a committed comet (the view shows a note instead).</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(IsHostVisible), nameof(IsWaitingForComet))]
  private bool _hasNucleusView;

  /// <summary>The camera is on the Earth → comet line (false after a drag, until the reset).</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(CaptionText))]
  [NotifyCanExecuteChangedFor(nameof(ResetToEarthCommand))]
  private bool _isEarthView = true;

  /// <summary>The date the view shows (the timeline's clock), as text; empty: unknown.</summary>
  [ObservableProperty]
  private string _dateText = string.Empty;

  /// <summary>The native view has an engine attached.</summary>
  public bool IsAttached => Viewport is { PresentationEngineId: not 0 };

  /// <summary>
  /// The native area is shown: until its engine is attached (a hidden host may never create its
  /// window) and then while there is a nucleus to look at. Otherwise the note is shown in its place.
  /// </summary>
  public bool IsHostVisible => !IsAttached || HasNucleusView;

  /// <summary>The engine is attached and no comet is committed yet.</summary>
  public bool IsWaitingForComet => IsAttached && !HasNucleusView;

  /// <summary>The line above the view: how it is oriented.</summary>
  public string CaptionText => IsEarthView ? StrCaptionEarth : StrCaptionFree;

  /// <summary>The engine was told to draw this view (tests).</summary>
  internal bool IsActive => _active;

  // ── Inputs from the panel ──────────────────────────────────────────────────

  /// <summary>A comet was committed or taken away: the view from Earth is installed again for the
  /// new one, a free view is dropped.</summary>
  public void SetCometCommitted(bool committed)
  {
    if (_cometCommitted == committed)
      return;
    _cometCommitted = committed;
    HasNucleusView = false;
    IsEarthView = true;
    Refresh();
  }

  /// <summary>The nucleus radius the view is framed for, km (≤ 0: a default).</summary>
  public void SetNucleusRadiusKm(double radiusKm)
  {
    if (double.IsNaN(radiusKm) || double.IsInfinity(radiusKm))
      radiusKm = 0;
    if (Math.Abs(radiusKm - _radiusKm) < 1e-9)
      return;
    _radiusKm = radiusKm;
    if (!IsAttached || !HasNucleusView)
      return;
    if (IsEarthView)
      InstallEarthView();
    else
    {
      ApplyProjection();
      PoseFreeView();
    }
  }

  private double RadiusKm => _radiusKm > 0 ? _radiusKm : DefaultRadiusKm;

  /// <summary>Camera to nucleus distance, AU: the nucleus spans
  /// 1 / <see cref="CameraService.NucleusFromEarthDiameters"/> of the view's height.</summary>
  internal double DistanceAu
  {
    get
    {
      double heightAu = 2.0 * RadiusKm * CameraService.NucleusFromEarthDiameters / AuToKm;
      return heightAu * 0.5 / Math.Tan(FovRad * 0.5);
    }
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  /// <summary>
  /// Brings the engine in line with what is shown: creates the native view's model on the first
  /// show, sets the camera up once its engine is attached, and pauses or resumes the view.
  /// </summary>
  public void Refresh()
  {
    if (_disposed)
      return;
    if (Viewport is null)
    {
      if (!IsShown)
        return;
      var viewport = _viewportFactory(OnPointer);
      viewport.Attached += OnAttached;
      viewport.MiniResized += OnResized;
      Viewport = viewport;
      // the view creates the native window from it; a model handed over already attached is set up now
    }
    if (!IsAttached)
      return;

    ulong pe = Viewport!.PresentationEngineId;
    if (IsShown)
    {
      // nothing is drawn before both flags are on the camera and the view from Earth is installed
      if (!_flagsSet)
        _flagsSet = SetFlags();
      if (_flagsSet && !HasNucleusView)
        InstallEarthView();
    }
    bool want = IsShown && _flagsSet && HasNucleusView;
    if (want != _active && _runtime.SetViewportActive(pe, want))
      _active = want;
  }

  private void OnAttached() => OnUi(() =>
  {
    _flagsSet = false;
    _active = false; // created paused
    OnPropertyChanged(nameof(IsAttached));
    OnPropertyChanged(nameof(IsHostVisible));
    OnPropertyChanged(nameof(IsWaitingForComet));
    Refresh();
  });

  private void OnResized(uint width, uint height) => OnUi(() =>
  {
    if (IsAttached && _flagsSet)
      ApplyProjection();
  });

  private float _exposureEv;

  /// <summary>The <see cref="ExposureEv"/> slider's step, stops.</summary>
  public const double ExposureEvStep = 0.5;

  /// <summary>Lowest / highest <see cref="ExposureEv"/> (the slider's range), stops.</summary>
  public static double MinExposureEv => CameraService.MinNucleusExposureEv;
  public static double MaxExposureEv => CameraService.MaxNucleusExposureEv;

  /// <summary>
  /// The mini-view's exposure, stops: 0 draws the point under the Sun white whatever the comet's
  /// distance from the Sun; each stop doubles the sunlit side's brightness, the night side stays
  /// black. Clamped to [<see cref="MinExposureEv"/>, <see cref="MaxExposureEv"/>].
  /// </summary>
  public float ExposureEv
  {
    get => _exposureEv;
    set
    {
      float v = float.IsNaN(value)
        ? 0f
        : Math.Max(CameraService.MinNucleusExposureEv, Math.Min(CameraService.MaxNucleusExposureEv, value));
      if (v == _exposureEv)
        return;
      _exposureEv = v;
      OnPropertyChanged();
      OnPropertyChanged(nameof(ExposureEvValue));
      OnPropertyChanged(nameof(ExposureEvText));
      ResetExposureCommand.NotifyCanExecuteChanged();
      SendExposure();
    }
  }

  /// <summary><see cref="ExposureEv"/> for a slider (double).</summary>
  public double ExposureEvValue
  {
    get => _exposureEv;
    set => ExposureEv = (float)value;
  }

  /// <summary><see cref="ExposureEv"/> as shown beside the slider ("+1.5 EV").</summary>
  public string ExposureEvText => string.Format(CultureInfo.InvariantCulture, "{0:+0.0;−0.0;0} EV", _exposureEv);

  /// <summary>Back to EV 0.</summary>
  [RelayCommand(CanExecute = nameof(CanResetExposure))]
  private void ResetExposure() => ExposureEv = 0f;

  private bool CanResetExposure() => _exposureEv != 0f;

  // the comet's distance from the Sun, AU (1 until known)
  private double _sunDistanceAu = 1.0;

  /// <summary>
  /// The comet's distance from the Sun, AU: the exposure takes out the sunlight's (1 AU / r)², so
  /// the view is as bright at 3 AU as at 1 AU. Resent when it changed by more than 1 %.
  /// </summary>
  internal void SetSunDistanceAu(double r)
  {
    if (double.IsNaN(r) || double.IsInfinity(r) || r <= 0 || Math.Abs(r / _sunDistanceAu - 1.0) <= 0.01)
      return;
    _sunDistanceAu = r;
    SendExposure();
  }

  /// <summary>
  /// The camera's sunlight factor at <paramref name="ev"/> for the dark-Lambert shading at
  /// <paramref name="sunDistanceAu"/>: 2^EV / albedo × r², which the shader's albedo and (1 AU / r)²
  /// bring back to 2^EV for the point under the Sun.
  /// </summary>
  internal static float ExposureFactor(float ev, double sunDistanceAu) =>
    (float)(CameraService.ExposureFor(CameraMode.CometOrbiting, physicalLighting: true, ev, nucleusFromEarth: true)
            * sunDistanceAu * sunDistanceAu);

  private void SendExposure()
  {
    if (_flagsSet && Viewport?.CameraEntityId is { } cam and not 0)
      _runtime.SetCameraExposure(cam, ExposureFactor(_exposureEv, _sunDistanceAu));
  }

  /// <summary>
  /// The dark Lambert shading (sunlight only, no ambient fill: the night side is black, so what is
  /// lit stands out), its brightness set by <see cref="ExposureEv"/> rather than by the physical
  /// irradiance: the distance from the Sun is taken out (<see cref="ExposureFactor"/>). This view
  /// is for finding the nucleus, its orientation and its lit side, not its photometry.
  /// </summary>
  private void ApplyShading(ulong cam)
  {
    _runtime.SetCameraShading(cam, CameraService.ShadingPhysical | CameraService.ShadingLambertDark);
    _runtime.SetCameraExposure(cam, ExposureFactor(_exposureEv, _sunDistanceAu));
  }

  /// <summary>No dust, no labels, display shading, 5 fps: what makes this view cheap and readable.</summary>
  private bool SetFlags()
  {
    ulong cam = Viewport?.CameraEntityId ?? 0;
    if (cam == 0)
      return false;
    bool noDust = _runtime.SetCameraNoDust(cam, true);
    bool noUi = _runtime.SetCameraNoUiLayer(cam, true);
    ApplyShading(cam);
    _runtime.SetViewportMinFrameInterval(Viewport!.PresentationEngineId, FrameIntervalUs);
    return noDust && noUi;
  }

  /// <summary>The perspective projection for the view's current size and distance.</summary>
  private bool ApplyProjection()
  {
    ulong cam = Viewport?.CameraEntityId ?? 0;
    if (cam == 0)
      return false;
    var (w, h) = Viewport!.MiniPixelSize;
    float aspect = w > 0 && h > 0 ? (float)w / h : 1f;
    double d = DistanceAu;
    return _runtime.CameraSetPerspective(cam, FovRad, aspect, (float)Math.Max(1e-12, d * 0.05), (float)(d * 200.0));
  }

  /// <summary>Hands the camera to the engine's view from Earth for this camera; it follows the
  /// comet and the clock from then on. False without a committed comet.</summary>
  private bool InstallEarthView()
  {
    ulong cam = Viewport?.CameraEntityId ?? 0;
    ulong earth = _runtime.EarthEntityId ?? 0;
    ulong comet = _cometCommitted ? _runtime.CometEntityId ?? 0 : 0;
    if (cam == 0 || earth == 0 || comet == 0)
    {
      HasNucleusView = false;
      return false;
    }
    _runtime.SetCameraParentToComet(cam, true);
    ApplyProjection();
    bool ok = _runtime.SetNucleusViewFromEarth(cam, earth, comet, DistanceAu);
    if (ok)
      IsEarthView = true;
    HasNucleusView = ok;
    return ok;
  }

  // ── Drag to orbit, reset ───────────────────────────────────────────────────

  // input thread → UI thread
  private void OnPointer(double x, double y, bool down, bool released) =>
    OnUi(() => HandlePointer(x, y, down, released));

  /// <summary>A pointer event over the native view: a held button drags, its release ends the drag.</summary>
  internal void HandlePointer(double x, double y, bool down, bool released)
  {
    if (released || !down)
    {
      _dragging = false;
      return;
    }
    if (!_dragging)
    {
      _dragging = true;
      _lastX = x;
      _lastY = y;
      return;
    }
    double dx = x - _lastX, dy = y - _lastY;
    _lastX = x;
    _lastY = y;
    if (dx != 0 || dy != 0)
      Orbit(dx, dy);
  }

  /// <summary>
  /// Orbits the camera about the nucleus by a drag of (<paramref name="dxPx"/>,
  /// <paramref name="dyPx"/>): the nucleus turns with the pointer. The first drag takes the camera
  /// from the engine's view from Earth, starting from that same place. False when there is nothing
  /// to orbit (no comet, or its direction from Earth is unknown).
  /// </summary>
  internal bool Orbit(double dxPx, double dyPx)
  {
    ulong cam = Viewport?.CameraEntityId ?? 0;
    if (cam == 0 || !_active || !HasNucleusView)
      return false;
    if (IsEarthView)
    {
      if (_runtime.CometEphemeris() is not { } eph)
        return false;
      const double RadToDeg = 180.0 / Math.PI;
      var toComet = SkyMath.EclipticDirection(eph.RaRad * RadToDeg, eph.DecRad * RadToDeg);
      _orbitDir = -toComet; // the camera stands on the Earth's side of the nucleus
      var north = CameraService.ObserverUpHint(ObserverUp.CelestialNorth, 0f);
      var up = north - _orbitDir * Vector3d.Dot(north, _orbitDir);
      _orbitUp = up.Length() > 1e-9 ? Vector3d.Normalize(up) : Vector3d.UnitZ;
      // the engine stops posing this camera
      _runtime.SetEarthObserver(cam, -1, 0, 0, false, 0, 0, 0, Quaterniond.Identity, ObserverUp.Ecliptic);
      IsEarthView = false;
    }

    // drag right: the camera goes to its left, the nucleus follows the pointer
    double yaw = dxPx * OrbitRadPerPixel, pitch = dyPx * OrbitRadPerPixel;
    var right = Vector3d.Normalize(Vector3d.Cross(_orbitUp, -_orbitDir));
    var dir = Vector3d.Normalize(_orbitDir * Math.Cos(yaw) - right * Math.Sin(yaw));
    // drag down: the camera goes up
    var up2 = Vector3d.Normalize(_orbitUp * Math.Cos(pitch) - dir * Math.Sin(pitch));
    dir = Vector3d.Normalize(dir * Math.Cos(pitch) + _orbitUp * Math.Sin(pitch));
    _orbitDir = dir;
    _orbitUp = Vector3d.Normalize(up2 - dir * Vector3d.Dot(up2, dir));
    return PoseFreeView();
  }

  /// <summary>Writes the free view's pose: at the orbit distance from the nucleus, looking at it.</summary>
  private bool PoseFreeView()
  {
    ulong cam = Viewport?.CameraEntityId ?? 0;
    ulong comet = _runtime.CometEntityId ?? 0;
    if (cam == 0 || comet == 0)
      return false;
    var offset = _orbitDir * DistanceAu;
    var rotation = CameraService.EarthObserverLookAt(-_orbitDir, _orbitUp);
    return _runtime.CameraSetRotoTranslate(cam, offset.X, offset.Y, offset.Z, rotation, comet);
  }

  private bool CanResetToEarth() => !IsEarthView;

  /// <summary>"Back to Earth view": the engine poses the camera on the Earth → comet line again.</summary>
  [RelayCommand(CanExecute = nameof(CanResetToEarth))]
  private void ResetToEarth()
  {
    _dragging = false;
    IsEarthView = true;
    if (IsAttached && _flagsSet)
      InstallEarthView();
    Refresh();
  }

  // ── Plumbing ───────────────────────────────────────────────────────────────

  private void OnUi(Action action)
  {
    if (_disposed)
      return;
    if (_dispatcher is null || _dispatcher.CheckAccess())
      action();
    else
      _dispatcher.Dispatch(() =>
      {
        if (!_disposed)
          action();
      });
  }

  /// <summary>Stops following the view. The engine itself is removed with the other viewports at
  /// shutdown (or by the native view when its window is destroyed).</summary>
  public void Dispose()
  {
    _disposed = true;
    _cultureSubscription?.Dispose();
    _positionSubscription?.Dispose();
    if (Viewport is { } viewport)
    {
      viewport.Attached -= OnAttached;
      viewport.MiniResized -= OnResized;
    }
  }
}
