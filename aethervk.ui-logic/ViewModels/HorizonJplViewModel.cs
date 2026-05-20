using System;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

public partial class HorizonJplViewModel : TabItemViewModel
{
  private readonly HorizonJplService _horizonService;
  private readonly ILocalStorageService _localStorage;
  private readonly BreadcrumbService _breadcrumb;

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

  public HorizonJplViewModel(HorizonJplService horizonService, ILocalStorageService localStorage, BreadcrumbService breadcrumb)
    : base("Horizon JPL")
  {
    _horizonService = horizonService;
    _localStorage = localStorage;
    _breadcrumb = breadcrumb;
  }

  [RelayCommand]
  private async Task SearchCometsAsync()
  {
    await _horizonService.FetchCometsAsync();
  }

  [RelayCommand]
  private void GoToStep2()
  {
    if (SelectedComet == null || SelectedComet.Length < 2)
      return;
    CurrentStep = 2;
    IsStep1Expanded = false;
    IsStep2Expanded = true;
  }

  [RelayCommand]
  private async Task SearchRecordsAsync()
  {
    if (SelectedComet == null || SelectedComet.Length < 2)
      return;
    var pdes = SelectedComet[1].Trim();
    string startStr = SearchStartTime?.ToString("yyyy-MM-dd") ?? "2024-01-01";
    string stopStr = SearchStopTime?.ToString("yyyy-MM-dd") ?? "2024-01-31";

    await _horizonService.FetchSpkRecor:wa:qadsAsync(pdes, startStr, stopStr);
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

  public ObservableCollection<string[]> ObjectData => _horizonService.ObjectData;

  [RelayCommand]
  private async Task DownloadObjectDataAsync()
  {
    if (SelectedSpkRecord == null)
      return;
    var spkId = SelectedSpkRecord[0].Trim();
    IsDownloading = true;
    await _horizonService.FetchObjectDataAsync(spkId);
    IsDownloading = false;
  }

  [RelayCommand]
  private async Task DownloadAndSaveSpkAsync()
  {
    if (SelectedComet == null || SelectedSpkRecord == null)
      return;

    var pdes = SelectedComet[1].Trim().Replace("/", "_").Replace(" ", "_");
    var spkId = SelectedSpkRecord[0].Trim();
    string startStr = SearchStartTime?.ToString("yyyy-MM-dd") ?? "2024-01-01";
    string stopStr = SearchStopTime?.ToString("yyyy-MM-dd") ?? "2024-01-31";

    var fileName = $"spk-kernels/{pdes}-{spkId}-{startStr}-{stopStr}.spk";
    var savePath = _localStorage.GetPersistentPath(fileName);

    IsDownloading = true;
    var result = await _horizonService.DownloadSpkByIdAsync(pdes, spkId, savePath, startStr, stopStr);
    IsDownloading = false;

    if (result != null)
    {
      await _breadcrumb.ShowMessageAsync("SPK Downloaded", $"Saved SPK to {fileName}");
    }
    else
    {
      await _breadcrumb.ShowMessageAsync("SPK Download Failed", "Could not download the SPK kernel.");
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
