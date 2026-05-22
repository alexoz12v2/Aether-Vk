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
        // Init native simulation engine with default scene
        _runtimeService.InitializeSimulationContext("Vulkan", null, true);
      });

      // Load ephemeris at startup
      await _runtimeService.LoadAlmanacFileAsync("assets/planets/de442.bsp");

      // Set time to Earth's 2020-01-01 position
      if (_runtimeService.ParseEpochToTaiSec("2020-01-01 00:00:00 UTC", out var taiSec))
      {
        ulong defaultSceneId = 1; // Since InitializeSimulationContext creates scene ID 1
        _runtimeService.SetSimulationTime(defaultSceneId, taiSec);
      }

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
