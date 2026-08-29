using System;
using System.Threading.Tasks;
using AetherVk.Logic.Services;

namespace AetherVk.Logic.ViewModels;

public partial class SplashViewModel : ViewModelBase
{
  public event Action? OnInitializationCompleted;
  public event Action<string>? OnInitializationFailed;

  public async Task InitializeAsync(Func<INativeRuntimeService> factory)
  {
    bool success = false;
    string errorMessage = "Unknown error";

    try
    {
      // Load ephemeris at startup concurrently
      // Now these are implicitly loaded in the constructor, during the native function if the
      // native function has a assets path available
      // _runtimeService.LoadAlmanacFileAsync("assets/planets/de442.bsp"),
      // _runtimeService.LoadAlmanacFileAsync("assets/earth_latest_high_prec.bpc")
      await Task.Run(() => factory.Invoke());

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
