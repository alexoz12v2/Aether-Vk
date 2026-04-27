using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

public partial class MeshViewerViewModel : TabItemViewModel
{
    private readonly NativeRuntimeService _runtimeService;
    private CancellationTokenSource? _cts;
    private Task? _lastRenderTask;
    private readonly bool _isLightTheme;

    public uint Width { get; } = 800;
    public uint Height { get; } = 600;

    [ObservableProperty]
    private bool _isInitialized;

    public event Action? OnFrameReady;

    public MeshViewerViewModel(string modelPath, string modelName, bool isLightTheme)
        : base(modelName)
    {
        _runtimeService = new NativeRuntimeService();
        _isLightTheme = isLightTheme;
        _ = InitializeSceneAsync(modelPath, modelName);
    }

    private async Task InitializeSceneAsync(string modelPath, string modelName)
    {
        if (IsInitialized) return;

        await Task.Run(() =>
        {
            try
            {
                if (!_runtimeService.IsInitialized)
                {
                    _runtimeService.InitializeSimulationContext("Vulkan", Width, Height, null, false);
                }
                if (_isLightTheme)
                {
                    _runtimeService.SetClearColor(1.0f, 1.0f, 1.0f, 1.0f);
                }
                else
                {
                    _runtimeService.SetClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                }

                ulong modelId = _runtimeService.ImportModel(modelPath);
                if (modelId > 0)
                {
                    _runtimeService.SpawnModelInstance(modelId, modelName);
                }
                
                var root = System.Linq.Enumerable.FirstOrDefault(_runtimeService.RootEntities);
                var camera = _runtimeService.CreateCamera(root);
                
                // Configure camera specifically for Mesh Viewer (like in the native test)
                var camTransform = System.Linq.Enumerable.FirstOrDefault(System.Linq.Enumerable.OfType<AetherVk.Logic.Models.TransformComponent>(camera.Components));
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
                
                _runtimeService.SetActiveCamera(camera.Id);
                var sun = _runtimeService.SpawnEntity("sun", root);
                sun.Components.Add(new AetherVk.Logic.Models.SunComponent());
            }
            catch (System.DllNotFoundException)
            {
                // Ignored for testing without vulkan
            }
        });

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
        Task.Run(async () =>
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

                    _runtimeService.SimulationTick();

                    if (_lastRenderTask != null)
                    {
                        await _lastRenderTask;
                        OnFrameReady?.Invoke();
                    }

                    _lastRenderTask = _runtimeService.RenderTickAsync();
                }
                else
                {
                    await Task.Delay(16, token);
                }
            }
        }, token);
    }

    public void CopyFrameToBuffer(IntPtr bufferPtr, nuint bufferSize)
    {
        _runtimeService.DownloadImage(bufferPtr, bufferSize);
    }

    public void Stop()
    {
        _cts?.Cancel();
        Task.Run(() => _runtimeService.ShutdownSimulation());
    }
}
