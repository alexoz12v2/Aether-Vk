using System;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

public partial class SplashViewModel : ViewModelBase
{
  private readonly NativeRuntimeService _runtimeService;

  public event Action? OnInitializationCompleted;
  public event Action<string>? OnInitializationFailed;

  public SplashViewModel(NativeRuntimeService runtimeService)
  {
    _runtimeService = runtimeService;
  }

  public async Task InitializeAsync()
  {
    bool success = false;
    string errorMessage = "Unknown error";

    try
    {
      await Task.Run(() =>
      {
        // Init native simulation engine
        _runtimeService.InitializeSimulationContext("Vulkan", null, false);

        // Create an empty scene
        ulong sceneId = _runtimeService.CreateScene(false);

        var root = _runtimeService.GetEntityByName(sceneId, "root");
        if (root != null)
        {
          _runtimeService.CreateSun(sceneId, root);
          _runtimeService.CreateSky(sceneId, root);
          _runtimeService.CreateCursor(sceneId, root);
          _runtimeService.CreateGrid(sceneId, root);
        }
      });

      success = true;
    }
    catch (Exception ex)
    {
      errorMessage = ex.Message;
    }

    if (success)
    {
      OnInitializationCompleted?.Invoke();
    }
    else
    {
      OnInitializationFailed?.Invoke(errorMessage);
    }
  }
}
