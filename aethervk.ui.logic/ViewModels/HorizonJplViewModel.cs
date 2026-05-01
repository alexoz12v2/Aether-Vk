using System;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

public class RequestSaveFileMessage
{
  public string DefaultFileName { get; }
  public TaskCompletionSource<string?> Result { get; } = new();

  public RequestSaveFileMessage(string defaultFileName) => DefaultFileName = defaultFileName;
}

public partial class HorizonJplViewModel : TabItemViewModel
{
  private readonly HorizonJplService _horizonService;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(IsStep2Enabled))]
  [NotifyPropertyChangedFor(nameof(IsStep3Enabled))]
  private int _currentStep = 1;

  [ObservableProperty]
  private bool _isStep1Expanded = true;

  [ObservableProperty]
  private bool _isStep2Expanded = false;

  [ObservableProperty]
  private bool _isStep3Expanded = false;

  public bool IsStep2Enabled => CurrentStep >= 2;
  public bool IsStep3Enabled => CurrentStep >= 3;

  [ObservableProperty]
  private DateTimeOffset? _searchStartTime = new DateTimeOffset(2024, 1, 1, 0, 0, 0, TimeSpan.Zero);

  [ObservableProperty]
  private DateTimeOffset? _searchStopTime = new DateTimeOffset(2024, 1, 31, 0, 0, 0, TimeSpan.Zero);

  public ObservableCollection<string[]> CometsData => _horizonService.CometsData;
  public ObservableCollection<string> CometsHeaders => _horizonService.CometsHeaders;

  [ObservableProperty]
  private string[]? _selectedComet;

  public ObservableCollection<string[]> SpkRecordsData => _horizonService.SpkRecordsData;
  public ObservableCollection<string> SpkRecordsHeaders => _horizonService.SpkRecordsHeaders;

  [ObservableProperty]
  private string[]? _selectedSpkRecord;

  [ObservableProperty]
  private bool _isDownloading;

  public HorizonJplViewModel(HorizonJplService horizonService)
    : base("Horizon JPL")
  {
    _horizonService = horizonService;
  }

  [RelayCommand]
  private async Task SearchCometsAsync()
  {
    string startStr = SearchStartTime?.ToString("yyyy-MM-dd") ?? "2024-01-01";
    string stopStr = SearchStopTime?.ToString("yyyy-MM-dd") ?? "2024-01-31";
    await _horizonService.FetchCometsAsync(startStr, stopStr);
  }

  [RelayCommand]
  private async Task GoToStep2Async()
  {
    if (SelectedComet == null || SelectedComet.Length < 2)
      return;
    var pdes = SelectedComet[1].Trim();
    string startStr = SearchStartTime?.ToString("yyyy-MM-dd") ?? "2024-01-01";
    string stopStr = SearchStopTime?.ToString("yyyy-MM-dd") ?? "2024-01-31";

    await _horizonService.FetchSpkRecordsAsync(pdes, startStr, stopStr);
    CurrentStep = 2;
    IsStep1Expanded = false;
    IsStep2Expanded = true;
  }

  [RelayCommand]
  private void GoToStep3()
  {
    if (SelectedSpkRecord != null)
    {
      CurrentStep = 3;
      IsStep2Expanded = false;
      IsStep3Expanded = true;
    }
  }

  [RelayCommand]
  private async Task DownloadAndSaveSpkAsync()
  {
    if (SelectedComet == null || SelectedSpkRecord == null)
      return;

    var pdes = SelectedComet[1].Trim();
    var spkId = SelectedSpkRecord[0].Trim();
    var defaultName = $"{pdes}_{spkId}.bsp";

    var msg = new RequestSaveFileMessage(defaultName);
    WeakReferenceMessenger.Default.Send(msg);

    var savePath = await msg.Result.Task;
    if (!string.IsNullOrEmpty(savePath))
    {
      IsDownloading = true;
      string startStr = SearchStartTime?.ToString("yyyy-MM-dd") ?? "2024-01-01";
      string stopStr = SearchStopTime?.ToString("yyyy-MM-dd") ?? "2024-01-31";
      await _horizonService.DownloadSpkByIdAsync(pdes, spkId, savePath!, startStr, stopStr);
      IsDownloading = false;
    }
  }

  [RelayCommand]
  private void ResetWizard()
  {
    CurrentStep = 1;
    SelectedComet = null;
    SelectedSpkRecord = null;
    IsStep1Expanded = true;
    IsStep2Expanded = false;
    IsStep3Expanded = false;
  }

  [RelayCommand]
  private void OpenDocumentation()
  {
    try
    {
      System.Diagnostics.Process.Start(
        new System.Diagnostics.ProcessStartInfo
        {
          FileName = "https://ssd.jpl.nasa.gov/horizons/manual.html",
          UseShellExecute = true,
        }
      );
    }
    catch (System.Exception)
    {
      // Ignore
    }
  }
}
