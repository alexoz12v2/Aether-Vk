using System;
using System.Reactive.Concurrency;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

public partial class ViewportSettingsViewModel : ObservableObject, IDisposable
{
  private readonly INativeRuntimeService _runtimeService;
  private readonly ISchedulerProvider _schedulerProvider;
  private readonly CameraService _cameraService;
  private readonly CompositeDisposable _disposables = new();

  public ulong CameraId { get; }
  public string ViewportName { get; }

  public System.Collections.Generic.IReadOnlyList<string> DistanceUnits { get; } =
    new[] { "Astronomical Units (AU)", "Kilometers (km)", "Meters (m)" };

  private const double AuToKm = 149_597_870.7;

  /// <summary>Display units per AU for <paramref name="unitIndex"/> (0 AU, 1 km, 2 m).</summary>
  public static double DistanceUnitFactor(int unitIndex) => unitIndex switch
  {
    1 => AuToKm,
    2 => AuToKm * 1000.0,
    _ => 1.0,
  };

  private double DistFactor => DistanceUnitFactor(OrthoUnitIndex);

  // ── Projection ────────────────────────────────────────────────────────────

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(IsOrthographic))]
  private bool _isPerspective = true;

  public bool IsOrthographic => !IsPerspective;

  [ObservableProperty]
  private double _perspFovDeg = 30.0;

  public System.Collections.Generic.IReadOnlyList<string> FovUnits { get; } =
    new[] { "deg", "arcmin", "arcsec" };

  [ObservableProperty]
  private int _fovUnitIndex = 0;

  partial void OnFovUnitIndexChanged(int value)
  {
    OnPropertyChanged(nameof(PerspFovDisplay));
    OnPropertyChanged(nameof(PerspFovStep));
    OnPropertyChanged(nameof(PerspFovMin));
    OnPropertyChanged(nameof(PerspFovMax));
  }

  public double PerspFovDisplay
  {
    get => FovUnitIndex switch
    {
      1 => Math.Round(PerspFovDeg * 60.0, 6),
      2 => Math.Round(PerspFovDeg * 3600.0, 6),
      _ => Math.Round(PerspFovDeg, 6)
    };
    set => PerspFovDeg = FovUnitIndex switch
    {
      1 => value / 60.0,
      2 => value / 3600.0,
      _ => value
    };
  }

  public double PerspFovMin => FovUnitIndex switch
  {
    1 => (1.0 / 60.0),
    2 => 1.0,
    _ => (1.0 / 3600.0)
  };

  public double PerspFovMax => FovUnitIndex switch
  {
    1 => 179.0 * 60.0,
    2 => 179.0 * 3600.0,
    _ => 179.0
  };

  public double PerspFovStep => FovUnitIndex switch
  {
    1 => 1.0,
    2 => 10.0,
    _ => 1.0
  };

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(PerspNearDisplay))]
  private double _perspNear = 0.0001;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(PerspFarDisplay))]
  private double _perspFar = 200.0;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(OrthoHalfWidthDisplay))]
  private double _orthoHalfWidth = 0.0155;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(OrthoHalfHeightDisplay))]
  private double _orthoHalfHeight = 0.0155;

  [ObservableProperty]
  private int _orthoUnitIndex = 0;

  partial void OnOrthoUnitIndexChanged(int value)
  {
    OnPropertyChanged(nameof(OrthoHalfWidthDisplay));
    OnPropertyChanged(nameof(OrthoHalfHeightDisplay));
    OnPropertyChanged(nameof(OrthoNearDisplay));
    OnPropertyChanged(nameof(OrthoFarDisplay));
    OnPropertyChanged(nameof(PerspNearDisplay));
    OnPropertyChanged(nameof(PerspFarDisplay));
    OnPropertyChanged(nameof(OrthoExtentMin));
    OnPropertyChanged(nameof(OrthoExtentMax));
    OnPropertyChanged(nameof(OrthoExtentStep));
  }

  // Distances are stored canonically in AU and only converted for display: no rounding here, a
  // km-sized extent is ~1e-8 AU and Math.Round(x, 6) used to turn it into 0.

  public double OrthoHalfWidthDisplay
  {
    get => OrthoHalfWidth * DistFactor;
    set => OrthoHalfWidth = value / DistFactor;
  }

  public double OrthoHalfHeightDisplay
  {
    get => OrthoHalfHeight * DistFactor;
    set => OrthoHalfHeight = value / DistFactor;
  }

  public double OrthoNearDisplay
  {
    get => OrthoNear * DistFactor;
    set => OrthoNear = value / DistFactor;
  }

  public double OrthoFarDisplay
  {
    get => OrthoFar * DistFactor;
    set => OrthoFar = value / DistFactor;
  }

  public double PerspNearDisplay
  {
    get => PerspNear * DistFactor;
    set => PerspNear = value / DistFactor;
  }

  public double PerspFarDisplay
  {
    get => PerspFar * DistFactor;
    set => PerspFar = value / DistFactor;
  }

  /// <summary>Distance slider bounds in display units: 1e-12 AU (0.15 m) .. 1000 AU.</summary>
  public double OrthoExtentMin => 1e-12 * DistFactor;
  public double OrthoExtentMax => 1000.0 * DistFactor;
  public double OrthoExtentStep => 0.001 * DistFactor;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(OrthoNearDisplay))]
  private double _orthoNear = 0.0001;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(OrthoFarDisplay))]
  private double _orthoFar = 200.0;

  public bool IsOrthoProportionsLocked
  {
    get => _cameraService?.IsOrthoProportionsLocked ?? false;
    set
    {
      if (_cameraService != null && _cameraService.IsOrthoProportionsLocked != value)
      {
        _cameraService.IsOrthoProportionsLocked = value;
        OnPropertyChanged();
        if (value)
        {
          RestoreOrthoProportions();
        }
        RestoreOrthoProportionsCommand.NotifyCanExecuteChanged();
      }
    }
  }

  private bool _isUpdatingFromRuntime = false;
  private float _aspectRatio = 1f;
  private IDisposable? _projectionListenerToken;

  // ── Earth Observer Mode ───────────────────────────────────────────────────

  /// <summary>True when the camera is in Earth Observer mode (EarthPosition).</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(IsEarthObserverMode), nameof(IsUpZenithMode))]
  private CameraMode _currentCameraMode = CameraMode.UpZenith;

  public bool IsEarthObserverMode => CurrentCameraMode == CameraMode.EarthPosition;

  /// <summary>True in UpZenith mode: shows the "snap above" buttons.</summary>
  public bool IsUpZenithMode => CurrentCameraMode == CameraMode.UpZenith;

  /// <summary>Whether a comet is committed (enables the comet snap target).</summary>
  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(SnapAboveCometCommand))]
  private bool _isCometCommitted;

  [RelayCommand]
  private void SnapAboveSun() => _cameraService.SnapAbove(SnapTarget.Sun);

  [RelayCommand(CanExecute = nameof(IsCometCommitted))]
  private void SnapAboveComet() => _cameraService.SnapAbove(SnapTarget.Comet);

  [RelayCommand]
  private void SnapAboveEarth() => _cameraService.SnapAbove(SnapTarget.Earth);

  /// <summary>Observer latitude in degrees (−90 … +90). Writes to <see cref="CameraService"/>.</summary>
  [ObservableProperty]
  private double _earthObserverLatDeg = 0.0;

  /// <summary>Observer longitude in degrees (−180 … +180). Writes to <see cref="CameraService"/>.</summary>
  [ObservableProperty]
  private double _earthObserverLonDeg = 0.0;

  /// <summary>Current Earth Observer look-direction mode. Two-way bound to <see cref="CameraService"/>.</summary>
  [ObservableProperty]
  private EarthObserverOrientationMode _earthObserverOrientationMode =
    EarthObserverOrientationMode.Free;



#if DEBUG
  protected ViewportSettingsViewModel(string name, bool isPerspective)
  {
    _runtimeService = null!;
    _schedulerProvider = null!;
    _cameraService = null!;
    _isUpdatingFromRuntime = true;
    ViewportName = name;
    IsPerspective = isPerspective;
  }
#endif

  public ViewportSettingsViewModel(
    ulong cameraId,
    int index,
    INativeRuntimeService runtimeService,
    ISchedulerProvider schedulerProvider,
    CameraService cameraService
  )
  {
    _runtimeService = runtimeService;
    _schedulerProvider = schedulerProvider;
    _cameraService = cameraService;
    CameraId = cameraId;
    ViewportName = $"Viewport {index + 1}";

    _projectionListenerToken = _runtimeService.RegisterSimulationListener(
      cameraId,
      ComponentForeignId.CameraProjection,
      HandleProjectionCallback
    );

    // Track camera mode to show/hide the Earth Observer subsection.
    _cameraService
      .CameraModeChanged.ObserveOn(schedulerProvider.MainThread)
      .Subscribe(mode => CurrentCameraMode = mode)
      .AddDisposableTo(_disposables);

    _cameraService
      .CometCommitted.ObserveOn(schedulerProvider.MainThread)
      .Subscribe(committed => IsCometCommitted = committed)
      .AddDisposableTo(_disposables);

    // "Restore preset projection" reacts to any projection change (user or Rust side).
    _cameraService
      .CameraProjection
      .CombineLatest(_cameraService.EarthObserverPresetProjection,
        (proj, preset) => CameraService.DiffersFromPreset(proj, preset))
      .DistinctUntilChanged()
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(modified => IsEarthPresetProjectionModified = modified)
      .AddDisposableTo(_disposables);

    // Mirror orientation mode changes that originate from other callers (e.g. future keybindings).
    _cameraService
      .EarthObserverOrientationModeChanged.ObserveOn(schedulerProvider.MainThread)
      .Subscribe(mode =>
      {
        // Suppress the OnChanged partial so we don't echo the change back to the service.
        _isUpdatingFromRuntime = true;
        try
        {
          EarthObserverOrientationMode = mode;
        }
        finally
        {
          _isUpdatingFromRuntime = false;
        }
      })
      .AddDisposableTo(_disposables);

    // Tracking below the horizon faces the horizon: explain it and suggest a site that sees the target.
    Observable
      .Timer(TimeSpan.Zero, TimeSpan.FromMilliseconds(500), schedulerProvider.Background)
      .Select(_ => _cameraService.GetEarthObserverTargetVisibility())
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(UpdateObserverHorizon)
      .AddDisposableTo(_disposables);

    _cameraService.ViewportResized += OnViewportResized;
  }

  private void OnViewportResized()
  {
    _schedulerProvider.MainThread.Schedule(() =>
    {
      RestoreOrthoProportionsCommand.NotifyCanExecuteChanged();
    });
  }

  private unsafe void HandleProjectionCallback(nint dataPtr)
  {
    var dto = *(CameraProjectionDTO*)dataPtr;
    _schedulerProvider.MainThread.Schedule(() =>
    {
      _isUpdatingFromRuntime = true;
      try
      {
        _aspectRatio = dto.Aspect;
        IsPerspective = dto.IsOrthographic == 0;
        if (IsPerspective)
        {
          PerspFovDeg = dto.Fov * 180.0 / Math.PI;
          PerspNear = dto.Near;
          PerspFar = dto.Far;
        }
        else
        {
          OrthoHalfWidth = dto.Right;
          OrthoHalfHeight = dto.Top;
          OrthoNear = dto.Near;
          OrthoFar = dto.Far;
        }
      }
      finally
      {
        _isUpdatingFromRuntime = false;
        RestoreOrthoProportionsCommand.NotifyCanExecuteChanged();
      }
    });
  }

  partial void OnPerspFovDegChanged(double value)
  {
      OnPropertyChanged(nameof(PerspFovDisplay));
      DispatchPerspective();
  }

  partial void OnPerspNearChanged(double value) => DispatchPerspective();

  partial void OnPerspFarChanged(double value) => DispatchPerspective();

  partial void OnOrthoHalfWidthChanged(double value)
  {
    if (!_isUpdatingFromRuntime && IsOrthoProportionsLocked)
    {
      _isUpdatingFromRuntime = true;
      OrthoHalfHeight = value / _cameraService.ViewportAspect;
      _isUpdatingFromRuntime = false;
    }
    DispatchOrthographic();
    RestoreOrthoProportionsCommand.NotifyCanExecuteChanged();
  }

  partial void OnOrthoHalfHeightChanged(double value)
  {
    if (!_isUpdatingFromRuntime && IsOrthoProportionsLocked)
    {
      _isUpdatingFromRuntime = true;
      OrthoHalfWidth = value * _cameraService.ViewportAspect;
      _isUpdatingFromRuntime = false;
    }
    DispatchOrthographic();
    RestoreOrthoProportionsCommand.NotifyCanExecuteChanged();
  }

  [RelayCommand(CanExecute = nameof(CanRestoreOrthoProportions))]
  private void RestoreOrthoProportions()
  {
    if (_isUpdatingFromRuntime)
      return;

    _isUpdatingFromRuntime = true;
    OrthoHalfWidth = OrthoHalfHeight * _cameraService.ViewportAspect;
    _isUpdatingFromRuntime = false;

    DispatchOrthographic();
    RestoreOrthoProportionsCommand.NotifyCanExecuteChanged();
  }

  private bool CanRestoreOrthoProportions()
  {
    if (IsPerspective || _cameraService == null)
      return false;
    if (Math.Abs(_cameraService.ViewportAspect) < 1e-5f)
      return false;

    double currentAspect = OrthoHalfWidth / OrthoHalfHeight;
    return Math.Abs(currentAspect - _cameraService.ViewportAspect) > 0.001;
  }

  partial void OnOrthoNearChanged(double value) => DispatchOrthographic();

  partial void OnOrthoFarChanged(double value) => DispatchOrthographic();

  partial void OnIsPerspectiveChanged(bool value)
  {
    if (_isUpdatingFromRuntime)
      return;
    if (value)
      DispatchPerspective();
    else
      // DispatchOrthographic handles the CometOrbiting branch internally, ensuring
      // formula-correct halfH (3×radius) and comet-appropriate near/far planes.
      DispatchOrthographic();
  }

  partial void OnEarthObserverLatDegChanged(double value)
  {
    if (_isUpdatingFromRuntime)
      return;
    _cameraService.SetEarthObserverLatLon((float)value, (float)EarthObserverLonDeg);
  }

  partial void OnEarthObserverLonDegChanged(double value)
  {
    if (_isUpdatingFromRuntime)
      return;
    _cameraService.SetEarthObserverLatLon((float)EarthObserverLatDeg, (float)value);
  }

  partial void OnEarthObserverOrientationModeChanged(EarthObserverOrientationMode value)
  {
    if (_isUpdatingFromRuntime)
      return;
    _cameraService.SetEarthObserverOrientationMode(value);
  }

  /// <summary>Enabled when the projection was changed away from the lock-in / tracking preset.</summary>
  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(RestoreEarthPresetProjectionCommand))]
  private bool _isEarthPresetProjectionModified;

  [RelayCommand(CanExecute = nameof(IsEarthPresetProjectionModified))]
  private void RestoreEarthPresetProjection() => _cameraService.RestoreEarthPresetProjection();

  /// <summary>True while a tracking target is below the observer's horizon (the Earth is in the way).</summary>
  [ObservableProperty]
  private bool _isObserverTargetBelowHorizon;

  /// <summary>Explains the below-horizon hold and suggests <see cref="SuggestedObserverLatDeg"/> /
  /// <see cref="SuggestedObserverLonDeg"/>.</summary>
  [ObservableProperty]
  private string _observerHorizonMessage = string.Empty;

  /// <summary>Latitude (°N) that has the tracked target at the zenith right now.</summary>
  [ObservableProperty]
  private double _suggestedObserverLatDeg;

  /// <summary>Longitude (°E) that has the tracked target at the zenith right now.</summary>
  [ObservableProperty]
  private double _suggestedObserverLonDeg;

  internal void UpdateObserverHorizon(EarthObserverTargetVisibility? v)
  {
    if (v is null || v.ElevationDeg >= 0.0)
    {
      IsObserverTargetBelowHorizon = false;
      return;
    }
    SuggestedObserverLatDeg = v.SuggestedLatDeg;
    SuggestedObserverLonDeg = v.SuggestedLonDeg;
    ObserverHorizonMessage = FormatObserverHorizonMessage(v);
    IsObserverTargetBelowHorizon = true;
  }

  internal static string FormatObserverHorizonMessage(EarthObserverTargetVisibility v)
  {
    var inv = System.Globalization.CultureInfo.InvariantCulture;
    string body = v.Target == EarthObserverTarget.Sun ? "Sun" : "comet";
    string lat = string.Format(inv, "{0:0.0}°{1}", Math.Abs(v.SuggestedLatDeg), v.SuggestedLatDeg >= 0.0 ? "N" : "S");
    string lon = string.Format(inv, "{0:0.0}°{1}", Math.Abs(v.SuggestedLonDeg), v.SuggestedLonDeg >= 0.0 ? "E" : "W");
    return string.Format(inv,
      "The {0} is {1:0.0}° below your horizon: the Earth is in the way, so tracking faces the horizon "
      + "where it will rise. It is overhead now at {2}, {3} (this spot drifts west ~15°/h as the Earth turns).",
      body, -v.ElevationDeg, lat, lon);
  }

  /// <summary>Moves the observer to the suggested site, where the tracked target is overhead.</summary>
  [RelayCommand]
  private void ApplySuggestedObserverSite()
  {
    double lat = SuggestedObserverLatDeg, lon = SuggestedObserverLonDeg;
    // one push with both coordinates (setting them one at a time would pose an intermediate site)
    _isUpdatingFromRuntime = true;
    try
    {
      EarthObserverLatDeg = lat;
      EarthObserverLonDeg = lon;
    }
    finally
    {
      _isUpdatingFromRuntime = false;
    }
    _cameraService.SetEarthObserverLatLon((float)lat, (float)lon);
    UpdateObserverHorizon(_cameraService.GetEarthObserverTargetVisibility());
  }

  private void DispatchPerspective()
  {
    if (_isUpdatingFromRuntime || !IsPerspective)
      return;
    // Use live viewport aspect ratio — _aspectRatio starts at 1.0 before the first Rust callback.
    float aspect = _cameraService.ViewportAspect;
    if (Math.Abs(aspect) < 1e-5f)
      aspect = 1f;

    _runtimeService.CameraSetPerspective(
      CameraId,
      (float)(PerspFovDeg * Math.PI / 180.0),
      aspect,
      (float)PerspNear,
      (float)PerspFar
    );
  }

  private void DispatchOrthographic()
  {
    if (_isUpdatingFromRuntime || IsPerspective)
      return;

    if (CurrentCameraMode == CameraMode.CometOrbiting)
    {
      var orbitOffset = _cameraService.GetOrbitOffset();
      float orbitMag  = orbitOffset.Length();
      float nearC = Math.Max(1e-12f, orbitMag * 0.05f);
      float farC  = orbitMag * 200f;

      _runtimeService.CameraSetOrthographic(
        CameraId,
        (float)-OrthoHalfWidth, (float)OrthoHalfWidth,
        (float)-OrthoHalfHeight, (float)OrthoHalfHeight,
        nearC, farC);
      return;
    }

    _runtimeService.CameraSetOrthographic(
      CameraId,
      (float)-OrthoHalfWidth,
      (float)OrthoHalfWidth,
      (float)-OrthoHalfHeight,
      (float)OrthoHalfHeight,
      (float)OrthoNear,
      (float)OrthoFar
    );
  }

  public void Dispose()
  {
    _cameraService.ViewportResized -= OnViewportResized;
    _projectionListenerToken?.Dispose();
    _disposables.Dispose();
  }
}
