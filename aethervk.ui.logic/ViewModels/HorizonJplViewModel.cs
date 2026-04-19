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
}
