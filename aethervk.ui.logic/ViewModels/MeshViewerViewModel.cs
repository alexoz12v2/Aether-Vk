using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

public partial class MeshViewerViewModel : TabItemViewModel, IDisposable
{
  private readonly NativeRuntimeService _runtimeService;
  private CancellationTokenSource? _cts;
  private Task<ulong>? _lastRenderTask;
  public ulong PresentationEngineId { get; private set; }
  private ulong _lastRenderTaskId;
  private readonly bool _isLightTheme;
  public ulong CameraId { get; private set; } = 1;
  public ulong SceneId { get; private set; }

  public uint Width { get; } = 800;
  public uint Height { get; } = 600;

  [ObservableProperty]
  private bool _isInitialized;

  private readonly ConsoleService? _consoleService;

  public event Action? OnFrameReady;

  public MeshViewerViewModel(ulong modelId, string modelPath, string modelName, bool isLightTheme, NativeRuntimeService runtimeService, ConsoleService? consoleService)
    : base(modelName)
  {
    _runtimeService = runtimeService;
    _consoleService = consoleService;
    _isLightTheme = isLightTheme;
    _ = InitializeSceneAsync(modelId, modelPath, modelName);
  }

  public void Dispose()
  {
    Stop();
    if (PresentationEngineId != 0)
    {
      _runtimeService.DestroyPresentationEngine(PresentationEngineId);
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

        _consoleService?.Log($"[MeshViewer] Checking PresentationEngineId...");
        if (PresentationEngineId == 0)
        {
          _consoleService?.Log($"[MeshViewer] Creating PresentationEngine...");
          PresentationEngineId = _runtimeService.CreatePresentationEngine(Width, Height);
        }

        _consoleService?.Log($"[MeshViewer] Creating Scene...");
        SceneId = _runtimeService.CreateScene(false);

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

        _consoleService?.Log($"[MeshViewer] Creating grid...");        var grid = _runtimeService.CreateGrid(SceneId, root);
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
    StartGameLoop();
  }

  public NativeRuntimeService RuntimeService => _runtimeService;

  private void StartGameLoop()
  {
    _cts = new CancellationTokenSource();
    var token = _cts.Token;

    // TODO: Correct render and simulation loop to be independent with each other, ie both
    // of them contain tasks with "generation sync"
    // Simulation Task i | Before | Render Task i
    // Simulation Task i | Before | Simulation Task i + 1
    // Render Task i | Before | Render Task i + 1
    Task.Run(
      async () =>
      {
        var sw = Stopwatch.StartNew();
        TimeSpan lastTime = sw.Elapsed;

        while (!token.IsCancellationRequested && _runtimeService.IsInitialized)
        {
          TimeSpan current = sw.Elapsed;
          TimeSpan dt = current - lastTime;

          if (dt.TotalMilliseconds >= 16.66)
          {
            lastTime = current;

            if (_lastRenderTask != null)
            {
              _lastRenderTaskId = await _lastRenderTask;
              OnFrameReady?.Invoke();
            }

            _lastRenderTask = _runtimeService.RenderTickAsync(
              PresentationEngineId,
              SceneId,
              CameraId,
              Width,
              Height
            );
          }
          else
          {
            await Task.Delay(16, token);
          }
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
