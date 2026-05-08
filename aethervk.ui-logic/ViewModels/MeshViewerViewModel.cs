using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using AetherVk.Logic.Input;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

public partial class MeshViewerViewModel
  : TabItemViewModel,
    IActionHandler,
    CommunityToolkit.Mvvm.Messaging.IRecipient<AetherVk.Logic.Messages.RenderFrameReadyMessage>,
    IDisposable
{
  private readonly NativeRuntimeService _runtimeService;
  public ulong PresentationEngineId { get; private set; }
  private ulong _lastRenderTaskId;
  private readonly bool _isLightTheme;
  public ulong CameraId { get; private set; } = 1;
  public ulong SceneId { get; private set; }

  public OperatorStack OperatorStack { get; }

  [ObservableProperty]
  private uint _width = 800;

  [ObservableProperty]
  private uint _height = 600;

  [ObservableProperty]
  private bool _isInitialized;

  private readonly ConsoleService? _consoleService;

  public event Action? OnFrameReady;

  public MeshViewerViewModel(
    ulong modelId,
    string modelPath,
    string modelName,
    bool isLightTheme,
    NativeRuntimeService runtimeService,
    ConsoleService? consoleService
  )
    : base(modelName)
  {
    _runtimeService = runtimeService;
    _consoleService = consoleService;
    _isLightTheme = isLightTheme;
    OperatorStack = new OperatorStack(new MeshViewerBaseOperator(this));
    _ = InitializeSceneAsync(modelId, modelPath, modelName);
  }

  public bool ProcessAction(AppAction action, bool isPressed)
  {
    return OperatorStack.ProcessAction(action, isPressed);
  }

  public bool ProcessPointerDelta(float dx, float dy) => OperatorStack.ProcessPointerDelta(dx, dy);

  public bool ProcessPointerWheel(float deltaY) => OperatorStack.ProcessPointerWheel(deltaY);

  public void Dispose()
  {
    Stop();
    if (PresentationEngineId != 0)
    {
      _runtimeService.DestroyPresentationEngine(SceneId, PresentationEngineId);
      PresentationEngineId = 0;
    }
  }

  private async Task InitializeSceneAsync(ulong modelId, string modelPath, string modelName)
  {
    if (IsInitialized)
      return;

    _consoleService?.Log($"[MeshViewer] Starting InitializeSceneAsync for {modelName}...");

    await Task.Run(async () =>
    {
      try
      {
        _consoleService?.Log($"[MeshViewer] Checking IsInitialized...");
        if (!_runtimeService.IsInitialized)
        {
          _consoleService?.Log($"[MeshViewer] Initializing Simulation Context...");
          _runtimeService.InitializeSimulationContext("Vulkan", null, false);
        }

        _consoleService?.Log($"[MeshViewer] Creating Scene...");
        SceneId = _runtimeService.CreateScene(false);
        _runtimeService.SetSceneDebugName(SceneId, $"MeshViewer_{modelName}");

        _consoleService?.Log($"[MeshViewer] Checking PresentationEngineId...");
        if (PresentationEngineId == 0)
        {
          _consoleService?.Log($"[MeshViewer] Creating PresentationEngine...");
          PresentationEngineId = _runtimeService.CreatePresentationEngine(Width, Height, SceneId);
        }

        if (modelId == 0)
        {
          _consoleService?.Log($"[MeshViewer] Importing Model (path={modelPath})...");
          modelId = await _runtimeService.ImportModelAsync(modelPath);
          _consoleService?.Log($"[MeshViewer] ImportModelAsync returned {modelId}");
        }

        if (modelId > 0)
        {
          _consoleService?.Log($"[MeshViewer] Spawning Model Instance...");
          await _runtimeService.SpawnModelInstanceAsync(SceneId, modelId, modelName);
        }

        _consoleService?.Log($"[MeshViewer] Getting root entity...");
        var root = _runtimeService.GetEntityByName(SceneId, "root");
        if (root == null)
        {
          _consoleService?.Log($"[MeshViewer] Root entity not found! Returning.");
          return;
        }

        _consoleService?.Log($"[MeshViewer] Creating camera...");
        var camera = _runtimeService.CreateCamera(SceneId, root);

        // Configure camera specifically for Mesh Viewer (like in the native test)
        var camTransform = System.Linq.Enumerable.FirstOrDefault(
          System.Linq.Enumerable.OfType<AetherVk.Logic.Models.TransformComponent>(camera.Components)
        );
        if (camTransform != null)
        {
          camTransform.PosX = 0.0f;
          camTransform.PosY = -5.0f;
          camTransform.PosZ = 0.0f;
          camTransform.RotW = 0.0f;
          camTransform.RotX = 0.0f;
          camTransform.RotY = 0.0f;
          camTransform.RotZ = 1.0f;
        }

        CameraId = camera.Id;

        _consoleService?.Log($"[MeshViewer] Creating sun...");
        var sun = _runtimeService.CreateSun(SceneId, root);
        var sunTransform = System.Linq.Enumerable.FirstOrDefault(
          System.Linq.Enumerable.OfType<AetherVk.Logic.Models.TransformComponent>(sun.Components)
        );
        if (sunTransform != null)
        {
          sunTransform.PosX = 0.0f;
          sunTransform.PosY = 10.0f; // Behind the camera (camera is at -5, looking at -Y)
          sunTransform.PosZ = 0.0f;
        }

        _consoleService?.Log($"[MeshViewer] Creating sky and cursor...");
        _runtimeService.CreateSky(SceneId, root);
        _runtimeService.CreateCursor(SceneId, root);

        _consoleService?.Log($"[MeshViewer] Creating grid...");
        var grid = _runtimeService.CreateGrid(SceneId, root);
        _consoleService?.Log($"[MeshViewer] Initialization complete!");
      }
      catch (System.Exception ex)
      {
        _consoleService?.Log($"[MeshViewer] Exception: {ex.Message}");
        // Ignored for testing without vulkan
      }
    });

    _consoleService?.Log($"[MeshViewer] Exiting InitializeSceneAsync!");
    IsInitialized = true;
    WeakReferenceMessenger.Default.Register<AetherVk.Logic.Messages.RenderFrameReadyMessage>(this, (r, m) => ((MeshViewerViewModel)r).Receive(m));
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
