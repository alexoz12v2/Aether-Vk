using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

public partial class Viewport3DViewModel
  : TabItemViewModel,
    IRecipient<AetherVk.Logic.Messages.ToggleAddJetModeMessage>
{
  private readonly NativeRuntimeService _runtimeService;
  private CancellationTokenSource? _cts;
  private Task<ulong>? _lastRenderTask;
  public ulong PresentationEngineId { get; private set; }
  private ulong _lastRenderTaskId;

  public uint Width { get; } = 800;
  public uint Height { get; } = 600;

  [ObservableProperty]
  private bool _isInitialized;

  [ObservableProperty]
  private bool _isLoading;

  [ObservableProperty]
  private bool _isAddingJet;

  [ObservableProperty]
  private bool _isMeasuringMode;

  [ObservableProperty]
  private bool _hasFirstMeasurementPoint;

  [ObservableProperty]
  private float _firstMeasurementPointX;

  [ObservableProperty]
  private float _firstMeasurementPointY;

  [ObservableProperty]
  private float _firstMeasurementPointZ;

  [ObservableProperty]
  private bool _showNoIntersectionFlyout;

  [ObservableProperty]
  private float _manualMeasurementX;

  [ObservableProperty]
  private float _manualMeasurementY;

  [ObservableProperty]
  private float _manualMeasurementZ;

  [ObservableProperty]
  private ulong _sceneId;

  public ulong CameraId { get; private set; } = 1;

  private static int _measurementCounter = 1;

  public event Action? OnFrameReady;

  public Viewport3DViewModel(NativeRuntimeService runtimeService)
    : base("Viewport 3D")
  {
    _runtimeService = runtimeService;
    _runtimeService.PropertyChanged += (s, e) =>
    {
      if (e.PropertyName == nameof(NativeRuntimeService.IsInitialized))
      {
        IsInitialized = _runtimeService.IsInitialized;
        if (IsInitialized)
        {
          SceneId = _runtimeService.CreateScene(true);
          StartGameLoop();
        }
      }
    };
    IsInitialized = _runtimeService.IsInitialized;
    CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Register(this);

    if (IsInitialized)
    {
      SceneId = _runtimeService.CreateScene(true);
      StartGameLoop();
    }
  }

  public void Receive(AetherVk.Logic.Messages.ToggleAddJetModeMessage message)
  {
    IsAddingJet = true;
  }

  public async void PerformRaycast(double x, double y, double w, double h)
  {
    float ndcX = (float)((x / w) * 2.0 - 1.0);
    float ndcY = (float)((y / h) * 2.0 - 1.0);

    var res = await _runtimeService.RaycastNdcAsync(SceneId, ndcX, ndcY);

    var breadcrumb =
      ServiceLocator.Provider?.GetService(typeof(BreadcrumbService)) as BreadcrumbService;

    if (IsMeasuringMode)
    {
      if (res.hit)
      {
        HandleMeasurementPoint(res.px, res.py, res.pz);
      }
      else
      {
        ShowNoIntersectionFlyout = true;
      }
      return;
    }

    if (res.hit)
    {
      var outlineVm =
        ServiceLocator.Provider?.GetService(typeof(OutlineViewModel)) as OutlineViewModel;
      var entity = _runtimeService.GetEntityById(SceneId, res.entityId);

      if (entity != null)
      {
        if (outlineVm?.SelectedEntity?.Id == entity.Id)
        {
          var comet = entity
            .Components.OfType<AetherVk.Logic.Models.CometComponent>()
            .FirstOrDefault();
          if (comet != null)
          {
            comet.Jets.Add(
              new AetherVk.Logic.Models.JetMarker
              {
                Name = "New Jet",
                PosX = res.px,
                PosY = res.py,
                PosZ = res.pz,
                ColorR = 1.0f,
                ColorG = 0.5f,
                ColorB = 0.0f,
                Size = 25000.0f, // Some visible scale
              }
            );
            breadcrumb?.ShowMessageAsync(
              "Raycast Hit",
              $"Placed new Jet on Comet at [{res.px:F1}, {res.py:F1}, {res.pz:F1}]"
            );
          }
          else
          {
            breadcrumb?.ShowMessageAsync(
              "Raycast Info",
              "Entity selected but it is not a comet, cannot add jets."
            );
          }
        }
        else
        {
          if (outlineVm != null)
          {
            outlineVm.SelectedEntity = entity;
            breadcrumb?.ShowMessageAsync("Raycast Hit", $"Selected {entity.Name}");
          }
        }
      }
    }
    else
    {
      // Deselect when clicking on empty space
      var outlineVm =
        ServiceLocator.Provider?.GetService(typeof(OutlineViewModel)) as OutlineViewModel;
      if (outlineVm != null)
      {
        outlineVm.SelectedEntity = null;
      }
    }
  }

  private void HandleMeasurementPoint(float x, float y, float z)
  {
    if (!HasFirstMeasurementPoint)
    {
      HasFirstMeasurementPoint = true;
      FirstMeasurementPointX = x;
      FirstMeasurementPointY = y;
      FirstMeasurementPointZ = z;
    }
    else
    {
      var name = $"Measurement_{_measurementCounter++}";
      _runtimeService.CreateMeasurement(
        SceneId,
        name,
        new[] { FirstMeasurementPointX, FirstMeasurementPointY, FirstMeasurementPointZ },
        new[] { x, y, z }
      );

      HasFirstMeasurementPoint = false;
      IsMeasuringMode = false;
      ShowNoIntersectionFlyout = false;
    }
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void SubmitManualMeasurement()
  {
    HandleMeasurementPoint(ManualMeasurementX, ManualMeasurementY, ManualMeasurementZ);
    ShowNoIntersectionFlyout = false;
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void SubmitCursorMeasurement()
  {
    float cx = 0,
      cy = 0,
      cz = 0;
    var state = ServiceLocator.Provider?.GetService(typeof(SceneStateManager)) as SceneStateManager;
    var rootEntities = state?.GetOrCreateScene(SceneId).RootEntities;
    var cursor = rootEntities?.FirstOrDefault(e =>
      e.Name == "cursor" || e.Components.Any(c => c.Name == "Cursor")
    );
    if (cursor != null)
    {
      var transform = cursor
        .Components.OfType<AetherVk.Logic.Models.TransformComponent>()
        .FirstOrDefault();
      if (transform != null)
      {
        cx = transform.PosX;
        cy = transform.PosY;
        cz = transform.PosZ;
      }
    }
    HandleMeasurementPoint(cx, cy, cz);
    ShowNoIntersectionFlyout = false;
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void UndoMeasurementRaycast()
  {
    ShowNoIntersectionFlyout = false;
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void ToggleMeasuringMode()
  {
    IsMeasuringMode = !IsMeasuringMode;
    HasFirstMeasurementPoint = false;
    ShowNoIntersectionFlyout = false;
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private async Task InitializeSceneAsync()
  {
    if (!_runtimeService.IsInitialized)
    {
      IsLoading = true;
      await Task.Run(() => _runtimeService.InitializeSimulationContext("Vulkan", null, false));
      IsLoading = false;
    }
    if (PresentationEngineId == 0)
    {
      PresentationEngineId = _runtimeService.CreatePresentationEngine(Width, Height);
    }
    SceneId = _runtimeService.CreateScene(true);
    IsInitialized = true;
    StartGameLoop();
  }

  public NativeRuntimeService RuntimeService => _runtimeService;

  private void StartGameLoop()
  {
    if (_cts != null)
      return;

    _cts = new CancellationTokenSource();
    var token = _cts.Token;

    Task.Run(
      async () =>
      {
        var sw = Stopwatch.StartNew();
        var lastTime = sw.Elapsed;
        var accumulatedTime = TimeSpan.Zero;
        var fixedTimeStep = TimeSpan.FromSeconds(1.0 / 60.0);

        while (!token.IsCancellationRequested)
        {
          var currentTime = sw.Elapsed;
          var deltaTime = currentTime - lastTime;
          lastTime = currentTime;
          accumulatedTime += deltaTime;

          if (IsInitialized)
          {
            // Fixed Update: Simulation stepping
            while (accumulatedTime >= fixedTimeStep)
            {
              // TODO: still to be implemented
              // _runtimeService.SimulationTick();
              accumulatedTime -= fixedTimeStep;
            }

            // Update camera
            var sceneState =
              ServiceLocator.Provider?.GetService(typeof(SceneStateManager)) as SceneStateManager;
            var camera = sceneState
              ?.GetOrCreateScene(SceneId)
              .EntityMap.Values.FirstOrDefault(e =>
                e.Name == "camera"
                || e.Components.Any(c => c is AetherVk.Logic.Models.CameraComponent)
              );
            if (camera != null)
            {
              CameraId = camera.Id;
            }

            // Wait for previous render to finish before starting a new one
            if (_lastRenderTask != null)
            {
              _lastRenderTaskId = await _lastRenderTask;
              OnFrameReady?.Invoke();
            }

            // Async Render - fire and forget, save task
            _lastRenderTask = _runtimeService.RenderTickAsync(
              PresentationEngineId,
              SceneId,
              CameraId,
              Width,
              Height
            );
          }

          // Yield to prevent pegging the CPU, aiming for ~60 FPS render signal
          await Task.Delay(16, token);
        }
      },
      token
    );
  }

  public async Task CopyFrameToBuffer(IntPtr bufferPtr, nuint bufferSize)
  {
    await _runtimeService.DownloadImageAsync(_lastRenderTaskId, bufferPtr, bufferSize);
  }

  public void Stop()
  {
    _cts?.Cancel();
  }
}
