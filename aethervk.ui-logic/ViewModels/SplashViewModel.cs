using System;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

public partial class SplashViewModel(INativeRuntimeService runtimeService) : ViewModelBase
{
  private readonly INativeRuntimeService _runtimeService = runtimeService;

  public event Action? OnInitializationCompleted;
  public event Action<string>? OnInitializationFailed;

  public async Task InitializeAsync()
  {
    bool success = false;
    string errorMessage = "Unknown error";

    if (!_runtimeService.Startup())
    {
      OnInitializationFailed?.Invoke("Startup Error");
      return;
    }

    try
    {
      // Load ephemeris at startup concurrently
      await Task.WhenAll(
        _runtimeService.LoadAlmanacFileAsync("assets/planets/de442.bsp"),
        _runtimeService.LoadAlmanacFileAsync("assets/earth_latest_high_prec.bpc")
      );
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
