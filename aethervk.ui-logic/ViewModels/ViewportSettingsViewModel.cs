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

  // ── Projection ────────────────────────────────────────────────────────────

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(IsOrthographic))]
  private bool _isPerspective = true;

  public bool IsOrthographic => !IsPerspective;

  [ObservableProperty]
  private double _perspFovDeg = 60.0;

  [ObservableProperty]
  private double _perspNear = 0.001;

  [ObservableProperty]
  private double _perspFar = 1000.0;

  [ObservableProperty]
  private double _orthoHalfWidth = 0.0155;

  [ObservableProperty]
  private double _orthoHalfHeight = 0.0155;

  [ObservableProperty]
  private double _orthoNear = 0.001;

  [ObservableProperty]
  private double _orthoFar = 1000.0;

  public bool IsOrthoProportionsLocked
  {
      get => _cameraService.IsOrthoProportionsLocked;
      set
      {
          if (_cameraService.IsOrthoProportionsLocked != value)
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
  private EarthObserverOrientationMode _earthObserverOrientationMode = EarthObserverOrientationMode.Inertial;

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
    _cameraService.CameraModeChanged
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(mode => CurrentCameraMode = mode)
      .AddDisposableTo(_disposables);

    // Mirror orientation mode changes that originate from other callers (e.g. future keybindings).
    _cameraService.EarthObserverOrientationModeChanged
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(mode =>
      {
        // Suppress the OnChanged partial so we don't echo the change back to the service.
        _isUpdatingFromRuntime = true;
        try { EarthObserverOrientationMode = mode; }
        finally { _isUpdatingFromRuntime = false; }
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

  partial void OnPerspFovDegChanged(double value) => DispatchPerspective();

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
      if (_isUpdatingFromRuntime) return;
      
      _isUpdatingFromRuntime = true;
      OrthoHalfWidth = OrthoHalfHeight * _cameraService.ViewportAspect;
      _isUpdatingFromRuntime = false;
      
      DispatchOrthographic();
      RestoreOrthoProportionsCommand.NotifyCanExecuteChanged();
  }

  private bool CanRestoreOrthoProportions()
  {
      if (IsPerspective) return false;
      if (Math.Abs(_cameraService.ViewportAspect) < 1e-5f) return false;
      
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
      DispatchOrthographic();
  }

  partial void OnEarthObserverLatDegChanged(double value)
  {
    if (_isUpdatingFromRuntime) return;
    _cameraService.SetEarthObserverLatLon((float)value, (float)EarthObserverLonDeg);
  }

  partial void OnEarthObserverLonDegChanged(double value)
  {
    if (_isUpdatingFromRuntime) return;
    _cameraService.SetEarthObserverLatLon((float)EarthObserverLatDeg, (float)value);
  }

  partial void OnEarthObserverOrientationModeChanged(EarthObserverOrientationMode value)
  {
    if (_isUpdatingFromRuntime) return;
    _cameraService.SetEarthObserverOrientationMode(value);
  }

  private void DispatchPerspective()
  {
    if (_isUpdatingFromRuntime || !IsPerspective)
      return;
    float aspect = _aspectRatio;
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

