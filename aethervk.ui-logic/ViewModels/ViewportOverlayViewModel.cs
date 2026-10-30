using System;
using System.Collections.ObjectModel;
using System.Numerics;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels.Debug;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

public partial class ViewportOverlayViewModel : ObservableObject, IDisposable
{
  private readonly CameraService _cameraService;
  private readonly INativeRuntimeService _runtimeService;
  private readonly BreadcrumbService _breadcrumbService;
  private readonly IUiThreadDispatcher _dispatcher;
  private readonly IFileDialogService _fileDialogService;
  private readonly ITabStateService<TimelineSession> _timelineSessionService;
  private readonly CometConfigService _cometConfigService;
  private readonly Viewport3DViewModel _viewportVm;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(FrameLabel))]
  private string _currentEpochString = string.Empty;

  /// <summary>Run to End in progress: the viewport spinner.</summary>
  [ObservableProperty]
  private bool _isRunningToEnd;

  [ObservableProperty]
  private string _jetPreviewSizeString = string.Empty;

  private readonly CompositeDisposable _disposables = [];
  private CameraProjectionState? _currentProjection;
  private double? _earthCometDistanceAu;

  // ── Camera Mode Badge ──────────────────────────────────────────────────────
  [ObservableProperty]
  private bool _isModeIndicatorExpanded;

  [ObservableProperty]
  [NotifyPropertyChangedFor(
    nameof(CameraModeName),
    nameof(IsModeEarthPosition),
    nameof(IsModeUpZenith),
    nameof(IsModeCometOrbiting),
    nameof(FrameLabel)
  )]
  private EarthObserverState _currentMode = EarthObserverState.UpZenith;

  /// <summary>The Earth observer's up, for <see cref="FrameLabel"/>.</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(FrameLabel))]
  private ObserverUp _observerUp = ObserverUp.Ecliptic;

  /// <summary>
  /// The reference frame the view is drawn in and the epoch shown (feedback #16), under the epoch:
  /// body-fixed / orbit plane / from Earth, which way is up, and at which date.
  /// </summary>
  public string FrameLabel => FrameLabelFor(CurrentMode, ObserverUp, CurrentEpochString);

  internal static string FrameLabelFor(EarthObserverState mode, ObserverUp up, string? epoch = null)
  {
    string frame = mode switch
    {
      EarthObserverState.UpZenith => "Orbit-plane frame (ecliptic J2000 seen from above, ecliptic north up)",
      EarthObserverState.CometOrbiting => "Body-fixed frame (turns with the nucleus, spin axis up)",
      EarthObserverState.EarthPositioning => up == ObserverUp.CelestialNorth
        ? "From Earth, celestial north up (J2000)"
        : "From Earth, ecliptic north up",
      _ => string.Empty,
    };
    return frame.Length == 0 || string.IsNullOrWhiteSpace(epoch) ? frame : $"{frame} · {epoch}";
  }

  // ── Mode buttons (feedback #20): Orbit / Nucleus / Model ─────────────────────
  [RelayCommand]
  private void SelectOrbitView() => _cameraService.SetCameraMode(CameraMode.UpZenith);

  [RelayCommand]
  private void SelectNucleusView() => _cameraService.SetCameraMode(CameraMode.CometOrbiting);

  [RelayCommand]
  private void SelectModelView() => _cameraService.EnterModelView();

  public string CameraModeName =>
    CurrentMode switch
    {
      EarthObserverState.EarthPositioning => "Earth Position",
      EarthObserverState.UpZenith => "Up Zenith",
      EarthObserverState.CometOrbiting => "Comet Orbiting",
      _ => string.Empty,
    };

  public bool IsModeEarthPosition => CurrentMode == EarthObserverState.EarthPositioning;
  public bool IsModeUpZenith => CurrentMode == EarthObserverState.UpZenith;
  public bool IsModeCometOrbiting => CurrentMode == EarthObserverState.CometOrbiting;


  // Measurement indicator
  [ObservableProperty]
  private string _measurementIndicatorText = "";

  [ObservableProperty]
  private double _measurementIndicatorWidth = 0.0;

  [ObservableProperty]
  private bool _showMeasurementIndicator = false;

  // Billboards
  public ObservableCollection<BillboardViewModel> Billboards { get; } = new();

#if DEBUG
  /// <summary>
  /// RenderDoc frame-capture sub-ViewModel. Non-null only in DEBUG builds.
  /// The overlay binds the button's <c>IsVisible</c> and <c>Command</c> here.
  /// </summary>
  public RenderDocCaptureViewModel? RenderDoc { get; }

  public DebugTelemetryPanelViewModel? DebugTelemetry { get; }
#else
  /// <summary>Always null in Release — ContentControl DataTemplate never fires.</summary>
  public object? RenderDoc => null;
  public object? DebugTelemetry => null;
#endif

  /// <summary>
  /// Camera global position/rotation tracker (all builds), polled every 500 ms.
  /// </summary>
  public CameraMatrixDebugViewModel CameraMatrixDebug { get; }

  /// <summary>
  /// Comet orbit debug panel ViewModel (all builds), refreshed at most every 100 ms.
  /// </summary>
  public CometOrbitDebugViewModel CometOrbitDebug { get; }

  private int _modeIndicatorChangeId;



  public ViewportOverlayViewModel(
    CameraService cameraService,
    INativeRuntimeService runtimeService,
    BreadcrumbService breadcrumbService,
    IUiThreadDispatcher dispatcher,
    IFileDialogService fileDialogService,
    ITabStateService<TimelineSession> timelineSessionService,
    CometConfigService cometConfigService,
    Viewport3DViewModel viewportVm,
    ISchedulerProvider schedulerProvider,
    TimelineService timelineService,
    ITabStateService<ModelSession> modelSessionService,
    PreferencesService? preferences = null
  )
  {
    _cameraService = cameraService;
    _runtimeService = runtimeService;
    _breadcrumbService = breadcrumbService;
    _dispatcher = dispatcher;
    _fileDialogService = fileDialogService;
    _timelineSessionService = timelineSessionService;
    _cometConfigService = cometConfigService;
    _viewportVm = viewportVm;
#if DEBUG
    RenderDoc        = new RenderDocCaptureViewModel(runtimeService, timelineService, modelSessionService, schedulerProvider);
    DebugTelemetry   = new DebugTelemetryPanelViewModel(runtimeService);
#endif
    CameraMatrixDebug = new CameraMatrixDebugViewModel(runtimeService, schedulerProvider, preferences,
      () => (_cameraService.GetCometScreenPosition(), _cameraService.ViewportWidthPx, _cameraService.ViewportHeightPx));
    CometOrbitDebug   = new CometOrbitDebugViewModel(preferences);
    _disposables.Add(CameraMatrixDebug);
    _disposables.Add(CometOrbitDebug);

    _cameraService
      .EarthObserverUpChanged.Subscribe(up => _dispatcher.Dispatch(() => ObserverUp = up))
      .AddDisposableTo(_disposables);

    _cameraService
      .CameraProjection.Subscribe(proj =>
      {
        _currentProjection = proj;
        UpdateMeasurementIndicator();
      })
      .AddDisposableTo(_disposables);

    // Sync Epoch
    Observable
      .Interval(TimeSpan.FromMilliseconds(100))
      .Subscribe(_ =>
      {
        _dispatcher.Dispatch(() =>
        {
          var targetId = new SessionId(typeof(TimelineSession), _viewportVm.SessionId.Number);
          var session = _timelineSessionService.GetSession(targetId);
          if (session != null && session.CurrentEpochString != CurrentEpochString)
          {
            CurrentEpochString = session.CurrentEpochString;
          }
          if (session != null && session.IsRunningToEnd != IsRunningToEnd)
          {
            IsRunningToEnd = session.IsRunningToEnd;
          }
        });
      })
      .AddDisposableTo(_disposables);

    // Sync Comet radius for Jet Preview string
    _cometConfigService
      .NucleusRadiusKm.Subscribe(r =>
      {
        // The Rust native runtime sets the jet preview sphere radius to exactly 1/50th of the nucleus radius.
        // radius_km / 50.0 = preview_radius_km
        // To display in meters, we multiply by 1000: (r / 50.0) * 1000 = r * 20.0 meters.
        JetPreviewSizeString = r > 0 ? $"Jet Preview Size: {r * 20.0:F1} m" : string.Empty;
        
        if (_currentProjection is { } p)
        {
          _dispatcher.DispatchAsync(() =>
          {
            if (_cameraService.CurrentMode == CameraMode.CometOrbiting)
            {
              UpdateOrbitDebugOrtho(p);
            }
            return Task.CompletedTask;
          });
        }
      })
      .AddDisposableTo(_disposables);

    _cameraService
      .CameraModeChanged.Subscribe(mode =>
      {
        CurrentMode = mode switch
        {
          CameraMode.EarthPosition => EarthObserverState.EarthPositioning,
          CameraMode.UpZenith => EarthObserverState.UpZenith,
          CameraMode.CometOrbiting => EarthObserverState.CometOrbiting,
          _ => EarthObserverState.UpZenith,
        };
        
        int changeId = ++_modeIndicatorChangeId;
        IsModeIndicatorExpanded = true;
        
        _ = Task.Delay(1800)
          .ContinueWith(_ => _dispatcher.Dispatch(() =>
          {
            if (_modeIndicatorChangeId == changeId)
              IsModeIndicatorExpanded = false;
          }));

        // Show/hide the comet orbit debug panel.
        CometOrbitDebug.IsVisible = mode == CameraMode.CometOrbiting;
      })
      .AddDisposableTo(_disposables);

    // ── Comet orbit debug panel subscription (10 Hz) ──────────────────────
    // Sample the authoritative camera transform (it arrives at the logic tick rate) and update
    // the debug panel with actual vs expected orbit distance and drift indicators.
    _cameraService
      .CameraTransform
      .Sample(OrbitDebugRefreshInterval, schedulerProvider.MainThread)
      .Subscribe(state =>
      {
        if (state == null || _cameraService.CurrentMode != CameraMode.CometOrbiting)
          return;
        UpdateOrbitDebugPanel(state);
      })
      .AddDisposableTo(_disposables);

    // the Earth observer's distance to the comet: the scale bar's length at the comet
    _cameraService
      .CameraTransform
      .Sample(OrbitDebugRefreshInterval, schedulerProvider.MainThread)
      .Subscribe(state =>
      {
        double? d = null;
        if (state != null && _cameraService.CurrentMode == CameraMode.EarthPosition
            && _cameraService.LastKnownCometPositionAu is { } c)
        {
          double dx = state.PosX - c.X, dy = state.PosY - c.Y, dz = state.PosZ - c.Z;
          d = Math.Sqrt(dx * dx + dy * dy + dz * dz);
        }
        bool changed = d is null != _earthCometDistanceAu is null
          || (d is { } a && _earthCometDistanceAu is { } b && Math.Abs(a - b) > 1e-4 * b);
        _earthCometDistanceAu = d;
        if (changed) UpdateMeasurementIndicator();
      })
      .AddDisposableTo(_disposables);

    _dispatcher.DispatchAsync(() =>
    {
      UpdateMeasurementIndicator();
      return Task.CompletedTask;
    });

    _cameraService
      .CameraProjection.Subscribe(proj =>
      {
        _currentProjection = proj;
        _dispatcher.DispatchAsync(() =>
        {
          UpdateMeasurementIndicator();
          if (_cameraService.CurrentMode == CameraMode.CometOrbiting && proj != null)
            UpdateOrbitDebugOrtho(proj);
          return Task.CompletedTask;
        });
      })
      .AddDisposableTo(_disposables);
  }

  public void Dispose()
  {
    _disposables.Dispose();
  }

  // ── Comet orbit debug panel update ────────────────────────────────────────

  private const double AuToKm = 149_597_870.7;

  /// <summary>Refresh period of the comet orbit debug panel (it is shown in Release too).</summary>
  public static readonly TimeSpan OrbitDebugRefreshInterval = TimeSpan.FromMilliseconds(100);

  private void UpdateOrbitDebugPanel(CameraTransformState state)
  {
    // Actual camera-to-comet distance (km), computed from authoritative transform.
    var comet = _cameraService.LastKnownCometPositionAu;
    double actualKm = 0.0;
    if (comet is { } c)
    {
      double dx = state.PosX - c.X;
      double dy = state.PosY - c.Y;
      double dz = state.PosZ - c.Z;
      actualKm = Math.Sqrt(dx * dx + dy * dy + dz * dz) * AuToKm;
    }

    // Expected distance from the spherical-coordinate orbit offset.
    Vector3 offset = _cameraService.GetOrbitOffset();
    double expectedKm = offset.Length() * AuToKm;
    double driftKm = Math.Abs(actualKm - expectedKm);

    CometOrbitDebug.ActualDistanceKm = $"dist actual:  {actualKm:F3} km";
    CometOrbitDebug.ExpectedDistanceKm = $"dist expect:  {expectedKm:F3} km";
    CometOrbitDebug.DriftKm = $"drift:        {driftKm:F4} km";
    CometOrbitDebug.DriftLevel =
      driftKm > 5.0 ? "error"
      : driftKm > 0.5 ? "warn"
      : "ok";

    CometOrbitDebug.OrbitAzimuthDeg = $"azimuth:      {_cameraService.OrbitAzimuthDeg:F2}°";
    CometOrbitDebug.OrbitElevationDeg = $"elevation:    {_cameraService.OrbitElevationDeg:F2}°";
    CometOrbitDebug.CameraOrientationQ =
      $"rot: ({state.RotX:F3}, {state.RotY:F3}, {state.RotZ:F3}, {state.RotW:F3})";
  }

  /// <summary>
  /// Updates the ortho-invariant row of the orbit debug panel.
  /// Compares the actual ortho half-height (from Rust-confirmed projection) against the
  /// expected value of 3 × nucleus radius, and sets the green/red indicator accordingly.
  /// </summary>
  private void UpdateOrbitDebugOrtho(CameraProjectionState proj)
  {
    if (proj.IsPerspective)
    {
      CometOrbitDebug.OrthoHalfHeightKm  = "perspective";
      CometOrbitDebug.OrthoExpectedKm    = "—";
      CometOrbitDebug.OrthoInvariantPassed = true;
      return;
    }

    double actualHalfH_km = (proj.Top - proj.Bottom) * 0.5 * AuToKm;

    float rKm = _cameraService.LastKnownNucleusRadiusKm > 0f
      ? _cameraService.LastKnownNucleusRadiusKm
      : 50f;
    double expectedHalfH_km = rKm * 3.0;

    double relErr = expectedHalfH_km > 0
      ? Math.Abs(actualHalfH_km - expectedHalfH_km) / expectedHalfH_km
      : 1.0;

    CometOrbitDebug.OrthoHalfHeightKm   = $"{actualHalfH_km:F3} km";
    CometOrbitDebug.OrthoExpectedKm     = $"{expectedHalfH_km:F3} km";
    CometOrbitDebug.OrthoInvariantPassed = relErr < 0.01; // 1% tolerance
  }


  [RelayCommand]
  private async Task InsertBillboard()
  {
    var filters = new[] { "png", "jpg", "jpeg", "bmp" };
    var path = await _fileDialogService.ShowOpenFileDialogAsync("Select Billboard Image", filters);
    if (!string.IsNullOrEmpty(path))
    {
      try
      {
        float ndcX = 0.5f;
        float ndcY = 0.5f;

        var entityId = _runtimeService.AddScreenSpaceBillboard(
          path!,
          new ScreenSpaceBillboard(
            NdcX: ndcX,
            NdcY: ndcY,
            Scale: 1.0f,
            RotationDeg: 0.0f,
            Opacity: 1.0f,
            ZIndex: 1
          )
        );

        if (entityId == 0)
        {
          _ = _breadcrumbService.ShowMessageAsync("Error", "Failed to create billboard entity.");
          return;
        }

        var billboard = new BillboardViewModel
        {
          EntityId = entityId,
          ImageSource = path,
          X = (_viewportVm.Width / 2.0) - 50,
          Y = (_viewportVm.Height / 2.0) - 50,
          Width = 100,
          Height = 100,
          ZIndex = 1,
          Opacity = 1.0,
          Scale = 1.0,
          Rotation = 0.0,
        };

        Billboards.Add(billboard);
        _ = _breadcrumbService.ShowMessageAsync(
          "Billboard Added",
          $"Loaded image {System.IO.Path.GetFileName(path)}"
        );
      }
      catch (Exception ex)
      {
        _ = _breadcrumbService.ShowMessageAsync("Error", $"Failed to load image: {ex.Message}");
      }
    }
  }

  [RelayCommand]
  private void RemoveBillboard(BillboardViewModel? billboard)
  {
    if (billboard == null)
      return;

    if (billboard.EntityId != 0)
    {
      _runtimeService.RemoveScreenSpaceBillboard(billboard.EntityId);
    }
    Billboards.Remove(billboard);
  }

  public void UpdateMeasurementIndicator()
  {
    if (_viewportVm.Width <= 0 || _viewportVm.Height <= 0 || _currentProjection == null)
    {
      ShowMeasurementIndicator = false;
      return;
    }
    double? atComet = _cameraService.CurrentMode == CameraMode.EarthPosition ? _earthCometDistanceAu : null;
    if (Measure(_currentProjection, _viewportVm.Width, atComet) is { } m)
      SetMeasurement(m.Nice, m.WidthPx, m.Unit, m.Suffix);
    else
      ShowMeasurementIndicator = false;
  }

  /// <summary>Lengths up to this are shown in km on the ortho scale bar, AU beyond (≈ 0.067 AU).</summary>
  internal const double MaxKmOnScaleBar = 1e7;

  /// <summary>
  /// The scale bar for <paramref name="proj"/> on a viewport <paramref name="widthPx"/> wide: a nice
  /// length (ortho: mm … km up to <see cref="MaxKmOnScaleBar"/>, then AU) or angle (perspective),
  /// the bar's width in px, and in perspective with <paramref name="targetDistanceAu"/> (the Earth
  /// observer's distance to the comet) the length the angle spans there. Null: degenerate.
  /// </summary>
  internal static (double Nice, double WidthPx, string Unit, string Suffix)? Measure(
    CameraProjectionState proj, double widthPx, double? targetDistanceAu)
  {
    if (!(widthPx > 0))
      return null;
    double targetPx = Math.Max(24.0, widthPx * 0.07);

    if (!proj.IsPerspective)
    {
      double wAu = proj.Right - proj.Left;
      if (!(wAu > 0))
        return null;
      double pxPerKm = widthPx / (wAu * AuToKm);
      double minKm = targetPx / pxPerKm;
      if (minKm >= MaxKmOnScaleBar)
      {
        double niceAu = GetNiceNumber(minKm / AuToKm);
        return (niceAu, niceAu * (widthPx / wAu), "AU", string.Empty);
      }
      if (minKm >= 1.0)
      {
        double niceKm = GetNiceNumber(minKm);
        return (niceKm, niceKm * pxPerKm, "km", string.Empty);
      }
      double minM = minKm * 1000.0;
      if (minM >= 1.0)
      {
        double niceM = GetNiceNumber(minM);
        return (niceM, niceM * pxPerKm / 1e3, "m", string.Empty);
      }
      if (minM >= 0.01)
      {
        double niceCm = GetNiceNumber(minM * 100.0);
        return (niceCm, niceCm * pxPerKm / 1e5, "cm", string.Empty);
      }
      double niceMm = GetNiceNumber(minM * 1000.0);
      return (niceMm, niceMm * pxPerKm / 1e6, "mm", string.Empty);
    }

    double wRad = 2.0 * Math.Atan(Math.Tan(proj.Fov / 2.0) * proj.Aspect);
    double wArcsec = wRad * 180.0 / Math.PI * 3600.0;
    if (!(wArcsec > 0))
      return null;
    double minArcsec = targetPx * (wArcsec / widthPx);
    double nice, arcsec;
    string unit;
    if (minArcsec >= 3600.0)
    {
      nice = GetNiceNumber(minArcsec / 3600.0);
      arcsec = nice * 3600.0;
      unit = "deg";
    }
    else if (minArcsec >= 60.0)
    {
      nice = GetNiceNumber(minArcsec / 60.0);
      arcsec = nice * 60.0;
      unit = "arcmin";
    }
    else
    {
      nice = GetNiceNumber(minArcsec);
      arcsec = nice;
      unit = "arcsec";
    }
    string suffix = string.Empty;
    if (targetDistanceAu is { } d && d > 0 && !double.IsInfinity(d))
    {
      double km = 2.0 * d * AuToKm * Math.Tan(arcsec / 3600.0 * Math.PI / 180.0 / 2.0);
      suffix = string.Concat(" ≈ ", FormatKm(km), " at the comet");
    }
    return (nice, arcsec * (widthPx / wArcsec), unit, suffix);
  }

  /// <summary>A length in km to about three significant digits ("61 300 km", "4.2 km").</summary>
  internal static string FormatKm(double km)
  {
    var ci = System.Globalization.CultureInfo.CurrentCulture;
    if (km >= 100.0)
    {
      double mag = Math.Pow(10, Math.Floor(Math.Log10(km)) - 2);
      return (Math.Round(km / mag) * mag).ToString("#,0", ci) + " km";
    }
    return km.ToString(km >= 10.0 ? "0.#" : "0.##", ci) + " km";
  }

  /// Shows the scale bar, or hides it when the projection produced a degenerate value
  /// (NaN/∞ bounds), so the label can never read "0 AU" or "NaN km".
  private void SetMeasurement(double nice, double widthPx, string unit, string suffix = "")
  {
    if (!(nice > 0) || double.IsInfinity(nice) || !(widthPx > 0) || double.IsInfinity(widthPx))
    {
      ShowMeasurementIndicator = false;
      return;
    }
    MeasurementIndicatorWidth = widthPx;
    MeasurementIndicatorText = $"{FormatNiceNumber(nice)} {unit}{suffix}";
    ShowMeasurementIndicator = true;
  }

  internal static double GetNiceNumber(double value)
  {
    if (value <= 0)
      return 1.0;
    double exponent = Math.Floor(Math.Log10(value));
    double fraction = value / Math.Pow(10, exponent);

    double niceFraction;
    if (fraction <= 1.0)
      niceFraction = 1.0;
    else if (fraction <= 2.0)
      niceFraction = 2.0;
    else if (fraction <= 5.0)
      niceFraction = 5.0;
    else
      niceFraction = 10.0;

    return niceFraction * Math.Pow(10, exponent);
  }

  internal static string FormatNiceNumber(double value)
  {
    if (value >= 1.0)
      return value.ToString("0");
    // value = m × 10^e with m ∈ {1,2,5}: -e decimals keep the leading digit, so it never prints "0".
    int decimals = (int)Math.Ceiling(-Math.Log10(value) - 1e-9);
    return value.ToString("F" + decimals);
  }
}
