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
