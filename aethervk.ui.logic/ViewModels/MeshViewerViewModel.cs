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
                    _runtimeService.InitializeSimulationContext("Vulkan", Width, Height);
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

        Task.Run(() =>
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
                    _ = _runtimeService.RenderTickAsync();
                    OnFrameReady?.Invoke();
                }
                else
                {
                    Thread.Sleep(1);
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
    }
}
