using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

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
      2 => PhysicsType == "Static" || PhysicsType == "Kinematic" || PhysicsType == "Dynamic",
      3 => FetchedOrbitData != null,
      4 => true,  // Spawn button is always enabled once the user reaches step 4
      _ => false,
    };

  // --- Step 1 ---
  public ObservableCollection<ImportedModelItem> ImportedModels { get; } = new();

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(CanGoNext))]
  private ImportedModelItem? _selectedModel;

  public bool HasNoModels => ImportedModels.Count == 0;
  public bool HasModels => !HasNoModels;

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
  private string _userLocalFrameString = "";

  [ObservableProperty]
  private string _simulationLocalFrameString = "";

  [ObservableProperty]
  private string _entityName = "New Comet";

  /// <summary>
  /// Comet nucleus radius in km, auto-populated from the Horizon JPL response.
  /// The user may edit this value manually before spawning.
  /// </summary>
  [ObservableProperty]
  private float _cometRadiusKm = 1.0f;

  /// <summary>Called automatically by the MVVM toolkit when FetchedOrbitData changes.</summary>
  partial void OnFetchedOrbitDataChanged(PlanetOrbitData? value)
  {
    if (value != null)
      CometRadiusKm = (float)value.CometRadiusKm;
    OnPropertyChanged(nameof(CanGoNext));
  }

  partial void OnSelectedModelChanged(ImportedModelItem? value)
  {
    OnPropertyChanged(nameof(CanGoNext));
    if (value != null)
    {
      if (value.RuntimeService.GetModelLocalFrames(value.Id, out var userFrame, out var simFrame))
      {
        UserLocalFrameString = $"[{userFrame.M00:F2}, {userFrame.M01:F2}, {userFrame.M02:F2}]\n" +
                               $"[{userFrame.M10:F2}, {userFrame.M11:F2}, {userFrame.M12:F2}]\n" +
                               $"[{userFrame.M20:F2}, {userFrame.M21:F2}, {userFrame.M22:F2}]";
        SimulationLocalFrameString = $"[{simFrame.M00:F2}, {simFrame.M01:F2}, {simFrame.M02:F2}]\n" +
                                     $"[{simFrame.M10:F2}, {simFrame.M11:F2}, {simFrame.M12:F2}]\n" +
                                     $"[{simFrame.M20:F2}, {simFrame.M21:F2}, {simFrame.M22:F2}]";
      }
    }
  }

  /// <summary>
  /// Converts the current Pitch/Yaw/Roll (degrees) Euler angles into a
  /// unit quaternion using the ZYX extrinsic convention (Roll→Yaw→Pitch).
  /// </summary>
  public (float w, float x, float y, float z) GetRotationQuaternion()
  {
    double pitchRad = Pitch * Math.PI / 180.0;
    double yawRad   = Yaw   * Math.PI / 180.0;
    double rollRad  = Roll  * Math.PI / 180.0;
    double cy = Math.Cos(yawRad   * 0.5), sy = Math.Sin(yawRad   * 0.5);
    double cp = Math.Cos(pitchRad * 0.5), sp = Math.Sin(pitchRad * 0.5);
    double cr = Math.Cos(rollRad  * 0.5), sr = Math.Sin(rollRad  * 0.5);
    double w = cr * cp * cy + sr * sp * sy;
    double x = sr * cp * cy - cr * sp * sy;
    double y = cr * sp * cy + sr * cp * sy;
    double z = cr * cp * sy - sr * sp * cy;
    return ((float)w, (float)x, (float)y, (float)z);
  }

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
  private async Task ImportMeshAsync()
  {
    var msg = new ImportModelRequestMessage();
    WeakReferenceMessenger.Default.Send(msg);
    var result = await msg.Response;
    if (result != null)
    {
      ImportedModels.Add(result);
      SelectedModel = result;
      OnPropertyChanged(nameof(HasNoModels));
      OnPropertyChanged(nameof(HasModels));
    }
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
