using System;
using System.Reactive.Concurrency;
using System.Reactive.Disposables;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

public partial class ViewportSettingsViewModel : ObservableObject, IDisposable
{
  private readonly INativeRuntimeService _runtimeService;
  private readonly ISchedulerProvider _schedulerProvider;
  private readonly CompositeDisposable _disposables = new();

  public ulong CameraId { get; }
  public string ViewportName { get; }

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

  private bool _isUpdatingFromRuntime = false;
  private float _aspectRatio = 1f;
  private IDisposable? _projectionListenerToken;

  public ViewportSettingsViewModel(
    ulong cameraId,
    int index,
    INativeRuntimeService runtimeService,
    ISchedulerProvider schedulerProvider
  )
  {
    _runtimeService = runtimeService;
    _schedulerProvider = schedulerProvider;
    CameraId = cameraId;
    ViewportName = $"Viewport {index + 1}";

    _projectionListenerToken = _runtimeService.RegisterSimulationListener(
      cameraId,
      ComponentForeignId.CameraProjection,
      HandleProjectionCallback
    );
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
      }
    });
  }

  partial void OnPerspFovDegChanged(double value) => DispatchPerspective();

  partial void OnPerspNearChanged(double value) => DispatchPerspective();

  partial void OnPerspFarChanged(double value) => DispatchPerspective();

  partial void OnOrthoHalfWidthChanged(double value) => DispatchOrthographic();

  partial void OnOrthoHalfHeightChanged(double value) => DispatchOrthographic();

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
    _projectionListenerToken?.Dispose();
    _disposables.Dispose();
  }
}
