using System;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

public partial class HorizonJplViewModel : TabItemViewModel
{
  private readonly HorizonJplService _horizonService;

  [ObservableProperty]
  private string _command = "499"; // Mars

  [ObservableProperty]
  private string _center = "500@399"; // Earth

  public ObservableCollection<string> CenterOptions { get; } = new() { "500@399", "@sun", "@ssb", "500@499" };

  [ObservableProperty]
  private string _startTime = "2024-01-01";

  [ObservableProperty]
  private string _stopTime = "2024-01-31";

  [ObservableProperty]
  private string _stepSize = "1 d";

  [ObservableProperty]
  private DateTimeOffset? _cometStartTime = new DateTimeOffset(2020, 1, 1, 0, 0, 0, TimeSpan.Zero);

  [ObservableProperty]
  private DateTimeOffset? _cometStopTime = new DateTimeOffset(2020, 12, 31, 0, 0, 0, TimeSpan.Zero);

  public ObservableCollection<string[]> Data => _horizonService.SessionData;
  public ObservableCollection<string> Headers => _horizonService.Headers;

  public ObservableCollection<string[]> CometsData => _horizonService.CometsData;
  public ObservableCollection<string> CometsHeaders => _horizonService.CometsHeaders;

  [ObservableProperty]
  private string[]? _selectedComet;

  partial void OnSelectedCometChanged(string[]? value)
  {
    if (value != null && value.Length > 0)
    {
      if (value.Length > 3 && !string.IsNullOrWhiteSpace(value[3]))
      {
        Command = $"DES={value[3].Trim()}; CAP";
      }
      else
      {
        string target = value[0];
        int startIdx = target.LastIndexOf('(');
        int endIdx = target.LastIndexOf(')');
        if (startIdx != -1 && endIdx != -1 && endIdx > startIdx)
        {
          Command = target.Substring(startIdx + 1, endIdx - startIdx - 1).Trim();
        }
        else
        {
          Command = target.Trim();
        }
      }
    }
  }

  public HorizonJplViewModel(HorizonJplService horizonService)
    : base("Horizon JPL")
  {
    _horizonService = horizonService;
  }

  [RelayCommand]
  private async Task FetchDataAsync()
  {
    await _horizonService.FetchDataAsync(Command, StartTime, StopTime, StepSize, Center);
  }

  [RelayCommand]
  private async Task FetchCometsAsync()
  {
    string startStr = CometStartTime?.ToString("yyyy-MM-dd") ?? "2020-01-01";
    string stopStr = CometStopTime?.ToString("yyyy-MM-dd") ?? "2020-12-31";
    await _horizonService.FetchCometsAsync(startStr, stopStr);
  }

  [RelayCommand]
  private void OpenDocumentation()
  {
    try
    {
      System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
      {
        FileName = "https://ssd.jpl.nasa.gov/horizons/manual.html",
        UseShellExecute = true
      });
    }
    catch (System.Exception)
    {
      // Ignore
    }
  }
}
