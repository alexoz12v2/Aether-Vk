using System;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
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
  private readonly TimelineService _timelineService;

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

  public ObservableCollection<CometSearchResult> CometsData => _horizonService.CometsData;
  public bool HasComets => CometsData.Count > 0;
  public bool HasNoComets => CometsData.Count == 0;

  [ObservableProperty]
  private CometSearchResult? _selectedComet;

  public ObservableCollection<SpkRecordItem> SpkRecordsData => _horizonService.SpkRecordsData;

  [ObservableProperty]
  private SpkRecordItem? _selectedSpkRecord;

  public ObservableCollection<ObjectDataProperty> ObjectData => _horizonService.ObjectData;

  [ObservableProperty]
  private bool _isDownloading;

  public HorizonJplViewModel(
    HorizonJplService horizonService,
    ILocalStorageService localStorage,
    BreadcrumbService breadcrumb,
    TimelineService timelineService
  )
    : base("Horizon JPL")
  {
    _horizonService = horizonService;
    _localStorage = localStorage;
    _breadcrumb = breadcrumb;
    _timelineService = timelineService;
  }

  [RelayCommand]
  private async Task SearchCometsAsync()
  {
    await _horizonService.FetchCometsAsync();
    OnPropertyChanged(nameof(HasComets));
    OnPropertyChanged(nameof(HasNoComets));
  }

  [RelayCommand]
  private void GoToStep2()
  {
    if (SelectedComet == null)
      return;
    CurrentStep = 2;
    IsStep1Expanded = false;
    IsStep2Expanded = true;
  }

  [RelayCommand]
  private async Task SearchRecordsAsync()
  {
    if (SelectedComet == null)
      return;

    var pdes = SelectedComet.PrimaryDesignation.Trim();
    string startStr = _timelineService.StartDate.ToString("yyyy-MM-dd");
    string stopStr = _timelineService.StopDate.ToString("yyyy-MM-dd");

    await _horizonService.FetchSpkRecordsAsync(pdes, startStr, stopStr);
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
  private async Task DownloadObservationAsync()
  {
    if (SelectedComet == null || SelectedSpkRecord == null)
      return;

    var pdes = SelectedComet.PrimaryDesignation.Trim().Replace("/", "_").Replace(" ", "_");
    var cmdId = SelectedSpkRecord.RecordId.Trim();  // JPL Horizons Record #, not NAIF SPK ID
    var start = _timelineService.StartDate;
    var stop = _timelineService.StopDate;

    var fileName =
      $"spk-kernels/{pdes}-{cmdId}-{start.ToString("yyyy-MM-dd")}-{stop.ToString("yyyy-MM-dd")}.spk";
    var savePath = _localStorage.GetPersistentPath(fileName);

    IsDownloading = true;
    var success = await _horizonService.DownloadObservationAsync(
      pdes,
      cmdId,
      start,
      stop,
      savePath
    );
    IsDownloading = false;
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
