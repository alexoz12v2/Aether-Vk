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

  public uint Width { get; } = 800;
  public uint Height { get; } = 600;

  [ObservableProperty]
  private bool _isInitialized;

  [ObservableProperty]
  private bool _isLoading;

  [ObservableProperty]
  private bool _isAddingJet;

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
          StartGameLoop();
        }
      }
    };
    IsInitialized = _runtimeService.IsInitialized;
    CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Register(this);

    if (IsInitialized)
    {
      StartGameLoop();
    }
  }

  public void Receive(AetherVk.Logic.Messages.ToggleAddJetModeMessage message)
  {
    IsAddingJet = true;
  }

  public async void PerformJetRaycast(double x, double y, double w, double h)
  {
    float ndcX = (float)((x / w) * 2.0 - 1.0);
    // Y axis points down in UI, but Vulkan expects NDC Y up or down depending on viewport
    // Our Viewport in Vulkan has negative height, meaning Y=0 is top, Y=height is bottom in screen space
    // So NDC Y is -1 at top, +1 at bottom (same as screen space mapping)
    float ndcY = (float)((y / h) * 2.0 - 1.0);

    var res = await _runtimeService.RaycastNdcAsync(ndcX, ndcY);

    var breadcrumb =
      ServiceLocator.Provider?.GetService(typeof(BreadcrumbService)) as BreadcrumbService;
    if (res.hit)
    {
      var entity = _runtimeService.GetEntityByName("12P/Pons-Brooks");
      if (entity != null && entity.Id == res.entityId)
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
      }
      else
      {
        breadcrumb?.ShowMessageAsync("Raycast Miss", "Hit a different entity.");
      }
    }
    else
    {
      breadcrumb?.ShowMessageAsync("Raycast Miss", "No intersection with physical mesh.");
    }
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private async Task InitializeSceneAsync()
  {
    if (!_runtimeService.IsInitialized)
    {
      IsLoading = true;
      await Task.Run(() => _runtimeService.InitializeSimulationContext("Vulkan", Width, Height));
      IsLoading = false;
    }
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
        TimeSpan lastTime = sw.Elapsed;

        while (!token.IsCancellationRequested)
        {
          TimeSpan current = sw.Elapsed;
          TimeSpan dt = current - lastTime;

          // ~60 FPS Target (16.66ms)
          if (dt.TotalMilliseconds >= 16.66 && IsInitialized)
          {
            lastTime = current;

            // Update camera
            ulong activeCam = _runtimeService.GetActiveCameraId();
            if (activeCam > 0)
            {
              _runtimeService.SetActiveCamera(activeCam);
            }

            // Render Frame Sync
            _runtimeService.RenderTickSync();

            // Notify View to copy frame
            OnFrameReady?.Invoke();
          }
          else
          {
            // Yield
            await Task.Delay(1, token);
          }
        }
      },
      token
    );
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
