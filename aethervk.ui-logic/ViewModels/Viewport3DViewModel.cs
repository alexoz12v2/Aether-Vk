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
    IRecipient<AetherVk.Logic.Messages.ToggleAddJetModeMessage>,
    IRecipient<AetherVk.Logic.Messages.RenderFrameReadyMessage>,
    IDisposable
{
  private readonly NativeRuntimeService _runtimeService;
  public ulong PresentationEngineId { get; private set; }
  private ulong _lastRenderTaskId;

  [ObservableProperty]
  private uint _width = 800;

  [ObservableProperty]
  private uint _height = 600;

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

  public ulong CameraId { get; private set; }

  private static int _measurementCounter = 1;

  public event Action? OnFrameReady;

  private readonly BreadcrumbService _breadcrumbService;
  private readonly SceneStateManager _sceneStateManager;

  private void SetupViewport()
  {
    var existingScene = _sceneStateManager.AllScenes.FirstOrDefault();
    SceneId = existingScene != null ? existingScene.SceneId : _runtimeService.CreateScene(true);
    
    if (PresentationEngineId == 0)
    {
      PresentationEngineId = _runtimeService.CreatePresentationEngine(Width, Height, SceneId);
    }
    
    var root = _runtimeService.GetEntityByName(SceneId, "root");
    if (root != null)
    {
        var camera = _runtimeService.CreateCamera(SceneId, root);
        CameraId = camera.Id;
    }
    else
    {
        CameraId = 1;
    }
  }

  public Viewport3DViewModel(
    NativeRuntimeService runtimeService,
    BreadcrumbService breadcrumbService,
    SceneStateManager sceneStateManager
  )
    : base("Viewport 3D")
  {
    _runtimeService = runtimeService;
    _breadcrumbService = breadcrumbService;
    _sceneStateManager = sceneStateManager;
    _runtimeService.PropertyChanged += (s, e) =>
    {
      if (e.PropertyName == nameof(NativeRuntimeService.IsInitialized))
      {
        IsInitialized = _runtimeService.IsInitialized;
        if (IsInitialized)
        {
          SetupViewport();
        }
      }
    };
    IsInitialized = _runtimeService.IsInitialized;
    WeakReferenceMessenger.Default.Register<AetherVk.Logic.Messages.RenderFrameReadyMessage>(this, (r, m) => ((Viewport3DViewModel)r).Receive(m));
    WeakReferenceMessenger.Default.Register<AetherVk.Logic.Messages.ToggleAddJetModeMessage>(this, (r, m) => ((Viewport3DViewModel)r).Receive(m));

    if (IsInitialized)
    {
      SetupViewport();
    }
  }

  public void Dispose()
  {
    Stop();
    if (PresentationEngineId != 0)
    {
      _runtimeService.DestroyPresentationEngine(SceneId, PresentationEngineId);
      PresentationEngineId = 0;
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

    var breadcrumb = _breadcrumbService;

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
      var state = _sceneStateManager.GetOrCreateScene(SceneId);
      var entity = _runtimeService.GetEntityById(SceneId, res.entityId);

      if (entity != null)
      {
        if (state.SelectedEntity?.Id == entity.Id)
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
          state.SelectedEntity = entity;
          CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(
            new AetherVk.Logic.ViewModels.EntitySelectedMessage(entity)
          );
          breadcrumb?.ShowMessageAsync("Raycast Hit", $"Selected {entity.Name}");
        }
      }
    }
    else
    {
      // Deselect when clicking on empty space
      var state = _sceneStateManager.GetOrCreateScene(SceneId);
      if (state.SelectedEntity != null)
      {
        state.SelectedEntity = null;
        CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(
          new AetherVk.Logic.ViewModels.EntitySelectedMessage(null)
        );
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
    var state = _sceneStateManager;
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
    
    SetupViewport();

    IsInitialized = true;
  }

  public NativeRuntimeService RuntimeService => _runtimeService;

  public void Receive(AetherVk.Logic.Messages.RenderFrameReadyMessage message)
  {
    if (message.PresentationEngineId == PresentationEngineId && message.SceneId == SceneId)
    {
      _lastRenderTaskId = message.RenderGeneration;
      OnFrameReady?.Invoke();
    }
  }

  public async Task CopyFrameToBuffer(IntPtr bufferPtr, nuint bufferSize)
  {
    await _runtimeService.DownloadImageAsync(_lastRenderTaskId, bufferPtr, bufferSize);
  }

  public void Stop() { }
}
