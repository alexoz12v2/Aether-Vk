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
  private string _currentEpochString = string.Empty;

  [ObservableProperty]
  private string _jetPreviewSizeString = string.Empty;

  private readonly CompositeDisposable _disposables = [];
  private CameraProjectionState? _currentProjection;

  // ── Camera Mode Badge ──────────────────────────────────────────────────────
  [ObservableProperty]
  private bool _isModeIndicatorExpanded;

  [ObservableProperty]
  [NotifyPropertyChangedFor(
    nameof(CameraModeName),
    nameof(IsModeEarthPosition),
    nameof(IsModeUpZenith),
    nameof(IsModeCometOrbiting)
  )]
  private EarthObserverState _currentMode = EarthObserverState.UpZenith;

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

  // ── Sun direction indicator ──────────────────────────────────────────────
  /// <summary>
  /// True when the sun is outside the camera frustum and the arrowhead should be shown.
  /// </summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(SunIndicatorLeft), nameof(SunIndicatorTop))]
  private bool _isSunOffScreen = false;

  /// <summary>
  /// Rotation angle of the arrowhead in degrees, measured clockwise from screen-up.
  /// 0° = sun is above centre, 90° = right, 180° = below, 270° = left.
  /// </summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(SunIndicatorLeft), nameof(SunIndicatorTop))]
  private double _sunIndicatorAngleDeg = 0.0;

  /// <summary>Radius of the indicator circle: min(32 epx, min(W,H)/2).</summary>
  private const double SunIndicatorMaxRadiusEpx = 32.0;

  private double ComputeIndicatorRadius() =>
    Math.Min(SunIndicatorMaxRadiusEpx, Math.Min(_viewportVm.Width, _viewportVm.Height) / 2.0);

  /// <summary>Canvas.Left of the arrowhead border (centred on the indicator circle point).</summary>
  public double SunIndicatorLeft
  {
    get
    {
      double r = ComputeIndicatorRadius();
      double rad = SunIndicatorAngleDeg * Math.PI / 180.0;
      return (_viewportVm.Width / 2.0) + r * Math.Sin(rad) - 12.0;
    }
  }

  /// <summary>Canvas.Top of the arrowhead border (centred on the indicator circle point).</summary>
  public double SunIndicatorTop
  {
    get
    {
      double r = ComputeIndicatorRadius();
      double rad = SunIndicatorAngleDeg * Math.PI / 180.0;
      return (_viewportVm.Height / 2.0) - r * Math.Cos(rad) - 12.0;
    }
  }

  // Billboards
  public ObservableCollection<BillboardViewModel> Billboards { get; } = new();

#if DEBUG
  /// <summary>
  /// RenderDoc frame-capture sub-ViewModel. Non-null only in DEBUG builds.
  /// The overlay binds the button's <c>IsVisible</c> and <c>Command</c> here.
  /// </summary>
  public RenderDocCaptureViewModel? RenderDoc { get; }

  public DebugTelemetryPanelViewModel? DebugTelemetry { get; }

  /// <summary>
  /// Camera matrix debug panel ViewModel. Non-null only in DEBUG builds.
  /// Shows the micro-layer view and projection matrices, throttled to ~5 fps.
  /// </summary>
  public CameraMatrixDebugViewModel? CameraMatrixDebug { get; }
#else
  /// <summary>Always null in Release — ContentControl DataTemplate never fires.</summary>
  public object? RenderDoc => null;
  public object? DebugTelemetry => null;
#endif

#if DEBUG
  /// <summary>
  /// Comet orbit debug panel ViewModel. Non-null only in DEBUG builds.
  /// </summary>
  public CometOrbitDebugViewModel? CometOrbitDebug { get; }
#else
  public object? CometOrbitDebug => null;
#endif

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
    ISchedulerProvider schedulerProvider
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
    RenderDoc        = new RenderDocCaptureViewModel(runtimeService);
    DebugTelemetry   = new DebugTelemetryPanelViewModel(runtimeService);
    CameraMatrixDebug = new CameraMatrixDebugViewModel(runtimeService, schedulerProvider);
    CometOrbitDebug   = new CometOrbitDebugViewModel();
#endif

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
        });
      })
      .AddDisposableTo(_disposables);

    // Sync Comet radius for Jet Preview string
    _cometConfigService
      .NucleusRadiusKm.Subscribe(r =>
      {
        // The MeshScaleMultiplierComponent multiplies the 1-meter base radius by the comet radius (in km).
        // So if nucleus is 2km, the jet mesh is scaled by 2000 compared to 1 meter? Wait!
        // The FFI sets: multiplier = ps_dto.nucleus_radius_km
        // The base mesh is 0.001 (1 meter)
        // Wait, 0.001 is 1 meter? Yes, 0.001 km = 1 meter.
        // If multiplier = nucleus_radius_km, then scale is 0.001 * 2 = 0.002 km = 2 meters.
        // So the Jet Preview size is linearly proportional to the comet radius.
        // A 1km comet gives a 1 meter jet preview.
        // So Jet Preview Size = Nucleus Radius * 1 meter.
        JetPreviewSizeString = r > 0 ? $"Jet Preview Size: {r:F1} m" : string.Empty;
        
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

#if DEBUG
        // Show/hide the comet orbit debug panel.
        if (CometOrbitDebug != null)
        {
          CometOrbitDebug.IsVisible = mode == CameraMode.CometOrbiting;
        }
#endif
      })
      .AddDisposableTo(_disposables);

    // ── Comet orbit debug panel subscription (~10 Hz) ─────────────────────
    // Sample the authoritative camera transform and update the debug panel
    // with actual vs expected orbit distance and drift indicators.
    _cameraService
      .CameraTransform.Subscribe(state =>
      {
        if (state == null || _cameraService.CurrentMode != CameraMode.CometOrbiting)
          return;
        UpdateOrbitDebugPanel(state);
      })
      .AddDisposableTo(_disposables);

    _dispatcher.DispatchAsync(() =>
    {
      UpdateMeasurementIndicator();
      return Task.CompletedTask;
    });

    // ── Sun direction indicator subscription ─────────────────────────────
    _cameraService
      .SunVisibilityChanged.Subscribe(state =>
      {
        if (state.IsVisible)
        {
          // Sun re-entered the frustum — hide the arrowhead.
          IsSunOffScreen = false;
          return;
        }

        // NDC convention from Rust: ndc_x > 0 = right, ndc_y > 0 = above screen centre.
        // atan2(ndc_x, ndc_y) → 0° = up, 90° = right, 180° = down, 270° = left (clockwise).
        double angleDeg = Math.Atan2(state.NdcX, state.NdcY) * 180.0 / Math.PI;
        SunIndicatorAngleDeg = angleDeg;
        IsSunOffScreen = true;
      })
      .AddDisposableTo(_disposables);

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

  private void UpdateOrbitDebugPanel(CameraTransformState state)
  {
#if DEBUG
    if (CometOrbitDebug == null) return;
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
#endif
  }

  /// <summary>
  /// Updates the ortho-invariant row of the orbit debug panel.
  /// Compares the actual ortho half-height (from Rust-confirmed projection) against the
  /// expected value of 3 × nucleus radius, and sets the green/red indicator accordingly.
  /// </summary>
  private void UpdateOrbitDebugOrtho(CameraProjectionState proj)
  {
#if DEBUG
    if (CometOrbitDebug == null) return;
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
#endif
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

    double target_px_width = Math.Max(24.0, _viewportVm.Width * 0.07);

    if (!_currentProjection.IsPerspective)
    {
      double W_au = _currentProjection.Right - _currentProjection.Left;
      if (W_au > 0)
      {
        double min_au = target_px_width * (W_au / _viewportVm.Width);

        if (min_au < 1e-6)
        {
          double min_km = min_au * 1.495978707e8;
          if (min_km < 1.0)
          {
            double min_m = min_km * 1000.0;
            double nice_m = GetNiceNumber(min_m);
            MeasurementIndicatorWidth = nice_m * (_viewportVm.Width / (W_au * 1.495978707e11));
            MeasurementIndicatorText = $"{FormatNiceNumber(nice_m)} m";
          }
          else
          {
            double nice_km = GetNiceNumber(min_km);
            MeasurementIndicatorWidth = nice_km * (_viewportVm.Width / (W_au * 1.495978707e8));
            MeasurementIndicatorText = $"{FormatNiceNumber(nice_km)} km";
          }
        }
        else
        {
          double nice_au = GetNiceNumber(min_au);
          MeasurementIndicatorWidth = nice_au * (_viewportVm.Width / W_au);
          MeasurementIndicatorText = $"{FormatNiceNumber(nice_au)} AU";
        }
        ShowMeasurementIndicator = true;
      }
      else
      {
        ShowMeasurementIndicator = false;
      }
    }
    else
    {
      double W_rad =
        2.0 * Math.Atan(Math.Tan(_currentProjection.Fov / 2.0) * _currentProjection.Aspect);
      double W_arcsec = (W_rad * 180.0 / Math.PI) * 3600.0;

      if (W_arcsec > 0)
      {
        double min_arcsec = target_px_width * (W_arcsec / _viewportVm.Width);

        if (min_arcsec >= 3600.0)
        {
          double min_deg = min_arcsec / 3600.0;
          double nice_deg = GetNiceNumber(min_deg);
          MeasurementIndicatorWidth = nice_deg * 3600.0 * (_viewportVm.Width / W_arcsec);
          MeasurementIndicatorText = $"{FormatNiceNumber(nice_deg)} deg";
        }
        else if (min_arcsec >= 60.0)
        {
          double min_min = min_arcsec / 60.0;
          double nice_min = GetNiceNumber(min_min);
          MeasurementIndicatorWidth = nice_min * 60.0 * (_viewportVm.Width / W_arcsec);
          MeasurementIndicatorText = $"{FormatNiceNumber(nice_min)} arcmin";
        }
        else
        {
          double nice_arcsec = GetNiceNumber(min_arcsec);
          MeasurementIndicatorWidth = nice_arcsec * (_viewportVm.Width / W_arcsec);
          MeasurementIndicatorText = $"{FormatNiceNumber(nice_arcsec)} arcsec";
        }
        ShowMeasurementIndicator = true;
      }
      else
      {
        ShowMeasurementIndicator = false;
      }
    }
  }

  private double GetNiceNumber(double value)
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

  private string FormatNiceNumber(double value)
  {
    if (value >= 1.0)
      return value.ToString("0");
    return value.ToString("0.#####");
  }
}
