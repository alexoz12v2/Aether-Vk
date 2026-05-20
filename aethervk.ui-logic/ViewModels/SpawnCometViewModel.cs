using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

public partial class SpawnCometViewModel : ObservableObject
{
  private readonly HorizonJplService _horizonService;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(CanGoNext))]
  [NotifyPropertyChangedFor(nameof(CanGoBack))]
  [NotifyPropertyChangedFor(nameof(IsStep1))]
  [NotifyPropertyChangedFor(nameof(IsStep2))]
  [NotifyPropertyChangedFor(nameof(IsStep3))]
  [NotifyPropertyChangedFor(nameof(IsStep4))]
  private int _currentStep = 1;

  public bool IsStep1 => CurrentStep == 1;
  public bool IsStep2 => CurrentStep == 2;
  public bool IsStep3 => CurrentStep == 3;
  public bool IsStep4 => CurrentStep == 4;

  public bool CanGoBack => CurrentStep > 1;

  public bool CanGoNext =>
    CurrentStep switch
    {
      1 => SelectedModel != null,
      2 => PhysicsType == "Static",
      3 => FetchedOrbitData != null,
      4 => false,
      _ => false,
    };

  // --- Step 1 ---
  public ObservableCollection<ImportedModelItem> ImportedModels { get; } = new();

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(CanGoNext))]
  private ImportedModelItem? _selectedModel;

  public bool HasNoModels => ImportedModels.Count == 0;

  // --- Step 2 ---
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(CanGoNext))]
  private string _physicsType = "Static"; // Static, Kinematic, Dynamic

  // --- Step 3 (Horizon Data) ---
  public ObservableCollection<string[]> CometsData => _horizonService.CometsData;
  public ObservableCollection<string> CometsHeaders => _horizonService.CometsHeaders;

  [ObservableProperty]
  private string[]? _selectedComet;

  [ObservableProperty]
  private DateTimeOffset? _targetDate = DateTimeOffset.Now;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(CanGoNext))]
  private PlanetOrbitData? _fetchedOrbitData;

  [ObservableProperty]
  private bool _isFetchingHorizonData;

  // --- Step 4 ---
  [ObservableProperty]
  private float _posX = 0f;

  [ObservableProperty]
  private float _posY = 0f;

  [ObservableProperty]
  private float _posZ = 0f;

  [ObservableProperty]
  private float _scaleX = 1f;

  [ObservableProperty]
  private float _scaleY = 1f;

  [ObservableProperty]
  private float _scaleZ = 1f;

  [ObservableProperty]
  private float _pitch = 0f;

  [ObservableProperty]
  private float _yaw = 0f;

  [ObservableProperty]
  private float _roll = 0f;

  [ObservableProperty]
  private string _entityName = "New Comet";

  public SpawnCometViewModel(
    IEnumerable<ImportedModelItem> models,
    HorizonJplService horizonService
  )
  {
    _horizonService = horizonService;
    foreach (var model in models)
    {
      ImportedModels.Add(model);
    }
    SelectedModel = ImportedModels.FirstOrDefault();
  }

  [RelayCommand]
  private async Task FetchCometsAsync()
  {
    IsFetchingHorizonData = true;
    await _horizonService.FetchCometsAsync();
    IsFetchingHorizonData = false;
  }

  [RelayCommand]
  private async Task FetchOrbitDataAsync()
  {
    if (SelectedComet == null || SelectedComet.Length < 2 || TargetDate == null)
      return;

    IsFetchingHorizonData = true;
    var pdes = SelectedComet[1].Trim();

    FetchedOrbitData = await _horizonService.GetPlanetDataAsync(pdes, TargetDate.Value.DateTime);
    IsFetchingHorizonData = false;
  }

  [RelayCommand]
  private void NextStep()
  {
    if (CanGoNext)
    {
      CurrentStep++;
    }
  }

  [RelayCommand]
  private void PreviousStep()
  {
    if (CanGoBack)
    {
      CurrentStep--;
    }
  }
}
