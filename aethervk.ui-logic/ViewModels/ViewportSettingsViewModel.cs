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
    new[] { "Astronomical Units (AU)", "Kilometers (km)" };

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
  private double _perspNear = 0.001;

  [ObservableProperty]
  private double _perspFar = 1000.0;

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
    OnPropertyChanged(nameof(OrthoExtentMin));
    OnPropertyChanged(nameof(OrthoExtentMax));
    OnPropertyChanged(nameof(OrthoExtentStep));
  }

  public double OrthoHalfWidthDisplay
  {
    get =>
      OrthoUnitIndex == 1
        ? Math.Round(OrthoHalfWidth * 149597870.7, 6)
        : Math.Round(OrthoHalfWidth, 6);
    set => OrthoHalfWidth = OrthoUnitIndex == 1 ? value / 149597870.7 : value;
  }

  public double OrthoHalfHeightDisplay
  {
    get =>
      OrthoUnitIndex == 1
        ? Math.Round(OrthoHalfHeight * 149597870.7, 6)
        : Math.Round(OrthoHalfHeight, 6);
    set => OrthoHalfHeight = OrthoUnitIndex == 1 ? value / 149597870.7 : value;
  }

  public double OrthoExtentMin => OrthoUnitIndex == 1 ? 1.0 : 0.00000001;
  public double OrthoExtentMax => OrthoUnitIndex == 1 ? 149597870700.0 : 1000.0;
  public double OrthoExtentStep => OrthoUnitIndex == 1 ? 100.0 : 0.001;

  [ObservableProperty]
  private double _orthoNear = 0.001;

  [ObservableProperty]
  private double _orthoFar = 1000.0;

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
  [NotifyPropertyChangedFor(nameof(IsEarthObserverMode))]
  private CameraMode _currentCameraMode = CameraMode.UpZenith;

  public bool IsEarthObserverMode => CurrentCameraMode == CameraMode.EarthPosition;

  /// <summary>Observer latitude in degrees (−90 … +90). Writes to <see cref="CameraService"/>.</summary>
  [ObservableProperty]
  private double _earthObserverLatDeg = 0.0;

  /// <summary>Observer longitude in degrees (−180 … +180). Writes to <see cref="CameraService"/>.</summary>
  [ObservableProperty]
  private double _earthObserverLonDeg = 0.0;

  /// <summary>Current Earth Observer look-direction mode. Two-way bound to <see cref="CameraService"/>.</summary>
  [ObservableProperty]
  private EarthObserverOrientationMode _earthObserverOrientationMode =
    EarthObserverOrientationMode.Inertial;



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
