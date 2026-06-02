using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

public partial class SpawnCometViewModel : ObservableObject
{
  private readonly HorizonJplService _horizonService;
  private readonly NativeRuntimeService _runtimeService;
  private readonly BreadcrumbService _breadcrumbService;

  // ── Step Navigation ────────────────────────────────────────────────────────

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(CanGoNext))]
  [NotifyPropertyChangedFor(nameof(CanGoBack))]
  [NotifyPropertyChangedFor(nameof(IsStep1))]
  [NotifyPropertyChangedFor(nameof(IsStep2))]
  [NotifyPropertyChangedFor(nameof(IsStep3))]
  [NotifyPropertyChangedFor(nameof(IsStep4))]
  [NotifyPropertyChangedFor(nameof(IsFinalStep))]
  private int _currentStep = 1;

  public bool IsStep1 => CurrentStep == 1;
  public bool IsStep2 => CurrentStep == 2;
  public bool IsStep3 => CurrentStep == 3;
  public bool IsStep4 => CurrentStep == 4;

  public bool IsFinalStep => IsStep4;
  public bool CanGoBack => CurrentStep > 1;

  public bool CanGoNext =>
    CurrentStep switch
    {
      1 => SelectedModel != null,
      2 => PhysicsType == "Static" || PhysicsType == "Kinematic",
      3 => HasValidSpkRecord,
      4 => IsTimelineValidated,
      _ => false,
    };

  // ── Step 1: Select Mesh ────────────────────────────────────────────────────

  public ObservableCollection<ImportedModelItem> ImportedModels { get; } = new();

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(CanGoNext))]
  private ImportedModelItem? _selectedModel;

  public bool HasNoModels => ImportedModels.Count == 0;
  public bool HasModels => !HasNoModels;

  // ── Step 2: Physics Type ───────────────────────────────────────────────────

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(CanGoNext))]
  [NotifyPropertyChangedFor(nameof(IsStaticMode))]
  [NotifyPropertyChangedFor(nameof(IsKinematicMode))]
  private string _physicsType = "Static"; // Static, Kinematic, Dynamic (WIP)

  public bool IsStaticMode => PhysicsType == "Static";
  public bool IsKinematicMode => PhysicsType == "Kinematic";

  [RelayCommand]
  private void SetPhysicsType(string type) => PhysicsType = type;

  // ── Static mode: Transform (position + rotation, NOT scale) ────────────────

  [ObservableProperty]
  private float _posX = 0f;

  [ObservableProperty]
  private float _posY = 0f;

  [ObservableProperty]
  private float _posZ = 0f;

  [ObservableProperty]
  private float _pitch = 0f;

  partial void OnPitchChanged(float value)
  {
    if (value >= 360f)
      Pitch = value % 360f;
    else if (value < 0f)
      Pitch = (value % 360f + 360f) % 360f;
  }

  [ObservableProperty]
  private float _yaw = 0f;

  partial void OnYawChanged(float value)
  {
    if (value >= 360f)
      Yaw = value % 360f;
    else if (value < 0f)
      Yaw = (value % 360f + 360f) % 360f;
  }

  [ObservableProperty]
  private float _roll = 0f;

  partial void OnRollChanged(float value)
  {
    if (value >= 360f)
      Roll = value % 360f;
    else if (value < 0f)
      Roll = (value % 360f + 360f) % 360f;
  }

  [ObservableProperty]
  private string _userLocalFrameString = "";

  [ObservableProperty]
  private string _simulationLocalFrameString = "";

  /// <summary>
  /// Converts the current Pitch/Yaw/Roll (degrees) Euler angles into a
  /// unit quaternion using the ZYX extrinsic convention (Roll→Yaw→Pitch).
  /// </summary>
  public (float w, float x, float y, float z) GetRotationQuaternion()
  {
    double pitchRad = Pitch * Math.PI / 180.0;
    double yawRad = Yaw * Math.PI / 180.0;
    double rollRad = Roll * Math.PI / 180.0;
    double cy = Math.Cos(yawRad * 0.5),
      sy = Math.Sin(yawRad * 0.5);
    double cp = Math.Cos(pitchRad * 0.5),
      sp = Math.Sin(pitchRad * 0.5);
    double cr = Math.Cos(rollRad * 0.5),
      sr = Math.Sin(rollRad * 0.5);
    double w = cr * cp * cy + sr * sp * sy;
    double x = sr * cp * cy - cr * sp * sy;
    double y = cr * sp * cy + sr * cp * sy;
    double z = cr * cp * sy - sr * sp * cy;
    return ((float)w, (float)x, (float)y, (float)z);
  }

  // ── Step 3: Choose Comet (JPL) + Nucleus Data ──────────────────────────────

  public ObservableCollection<CometSearchResult> CometsData => _horizonService.CometsData;
  public ObservableCollection<SpkRecordItem> SpkRecordsData => _horizonService.SpkRecordsData;

  [ObservableProperty]
  private CometSearchResult? _selectedComet;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(CanGoNext))]
  [NotifyPropertyChangedFor(nameof(HasValidSpkRecord))]
  private SpkRecordItem? _selectedSpkRecord;

  /// <summary>True when the selected SPK record is a real numeric id.</summary>
  public bool HasValidSpkRecord => SelectedSpkRecord?.IsValid == true;
  public bool HasSpkRecords => SpkRecordsData.Count > 0;
  public bool HasNoSpkRecords => !HasSpkRecords;

  [ObservableProperty]
  private int _orbitYear = DateTime.Now.Year;

  public bool HasComets => CometsData.Count > 0;
  public bool HasNoComets => CometsData.Count == 0;

  [ObservableProperty]
  private bool _isFetchingHorizonData;

  // Nucleus physical parameters
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(IsRadiusOverridden))]
  private float _cometRadiusKm = 1.0f;

  private float _jplRadiusKm = 1.0f;

  public bool IsRadiusOverridden => Math.Abs(CometRadiusKm - _jplRadiusKm) > 0.0005f;

  [ObservableProperty]
  private double _massKg = 1e13;

  // IAU Rotational Model parameters (from PhysicalMeshComponent)
  [ObservableProperty]
  private double _poleRaDeg = 0.0;

  [ObservableProperty]
  private double _poleDecDeg = 90.0;

  [ObservableProperty]
  private double _primeMeridianDeg = 0.0;

  [ObservableProperty]
  private double _poleRaRateDeg = 0.0;

  [ObservableProperty]
  private double _poleDecRateDeg = 0.0;

  /// <summary>Spin rate in degrees/day. For kinematic mode, this drives the rotation.</summary>
  [ObservableProperty]
  private double _rotationRateDeg = 0.0;

  // Angular velocity (rad/s) — user-settable for initial angular momentum
  [ObservableProperty]
  private float _angularVelX = 0.0f;

  [ObservableProperty]
  private float _angularVelY = 0.0f;

  [ObservableProperty]
  private float _angularVelZ = 0.0f;

  [ObservableProperty]
  private string _entityName = "New Comet";

  // ── Step 4: Timeline & Validation ──────────────────────────────────────────

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(IsEpochRangeValid))]
  [NotifyPropertyChangedFor(nameof(CanValidateTimeline))]
  private DateTimeOffset _wizardStartEpoch;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(IsEpochRangeValid))]
  [NotifyPropertyChangedFor(nameof(CanValidateTimeline))]
  private DateTimeOffset _wizardEndEpoch;

  /// <summary>True when start is strictly before end.</summary>
  public bool IsEpochRangeValid => WizardStartEpoch < WizardEndEpoch;

  /// <summary>Can click Validate only when range is valid and not already validated.</summary>
  public bool CanValidateTimeline => IsEpochRangeValid && !IsTimelineValidated && !IsValidatingTimeline;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(CanGoNext))]
  [NotifyPropertyChangedFor(nameof(CanValidateTimeline))]
  private bool _isTimelineValidated;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(CanValidateTimeline))]
  private bool _isValidatingTimeline;

  [ObservableProperty]
  private string _timelineValidationStatus = "";

  // Reset validation when epochs change
  partial void OnWizardStartEpochChanged(DateTimeOffset value)
  {
    IsTimelineValidated = false;
    TimelineValidationStatus = "";
  }

  partial void OnWizardEndEpochChanged(DateTimeOffset value)
  {
    IsTimelineValidated = false;
    TimelineValidationStatus = "";
  }

  partial void OnSelectedSpkRecordChanged(SpkRecordItem? value)
  {
    OnPropertyChanged(nameof(CanGoNext));
    OnPropertyChanged(nameof(HasValidSpkRecord));
  }

  partial void OnSelectedModelChanged(ImportedModelItem? value)
  {
    OnPropertyChanged(nameof(CanGoNext));
    if (value != null)
    {
      if (value.RuntimeService.GetModelLocalFrames(value.Id, out var userFrame, out var simFrame))
      {
        UserLocalFrameString =
          $"[{userFrame.M00:F2}, {userFrame.M01:F2}, {userFrame.M02:F2}]\n"
          + $"[{userFrame.M10:F2}, {userFrame.M11:F2}, {userFrame.M12:F2}]\n"
          + $"[{userFrame.M20:F2}, {userFrame.M21:F2}, {userFrame.M22:F2}]";
        SimulationLocalFrameString =
          $"[{simFrame.M00:F2}, {simFrame.M01:F2}, {simFrame.M02:F2}]\n"
          + $"[{simFrame.M10:F2}, {simFrame.M11:F2}, {simFrame.M12:F2}]\n"
          + $"[{simFrame.M20:F2}, {simFrame.M21:F2}, {simFrame.M22:F2}]";
      }
    }
  }

  // ── Constructor ────────────────────────────────────────────────────────────

  public SpawnCometViewModel(
    IEnumerable<ImportedModelItem> models,
    HorizonJplService horizonService,
    NativeRuntimeService runtimeService,
    TimelineService timelineService,
    BreadcrumbService breadcrumbService,
    ulong? preselectedModelId = null
  )
  {
    _horizonService = horizonService;
    _runtimeService = runtimeService;
    _breadcrumbService = breadcrumbService;

    // Initialize epoch range from current timeline service
    WizardStartEpoch = timelineService.StartDate;
    WizardEndEpoch = timelineService.StopDate;

    if (DateTimeOffset.TryParse(timelineService.UtcTime, out var dt))
    {
      _orbitYear = dt.Year;
    }

    foreach (var model in models)
    {
      ImportedModels.Add(model);
    }

    if (preselectedModelId.HasValue)
    {
      SelectedModel = ImportedModels.FirstOrDefault(m => m.Id == preselectedModelId.Value);
      if (SelectedModel != null)
      {
        CurrentStep = 2; // Skip to step 2
      }
    }
    else
    {
      SelectedModel = ImportedModels.FirstOrDefault();
    }
  }

  // ── Commands ───────────────────────────────────────────────────────────────

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
    OnPropertyChanged(nameof(HasComets));
    OnPropertyChanged(nameof(HasNoComets));
  }

  [RelayCommand]
  private async Task FetchSpkRecordsAsync()
  {
    if (SelectedComet == null)
      return;
    IsFetchingHorizonData = true;
    SelectedSpkRecord = null;
    var start = $"{OrbitYear - 5}-01-01";
    var stop = $"{OrbitYear + 5}-12-31";
    await _horizonService.FetchSpkRecordsAsync(SelectedComet.PrimaryDesignation, start, stop);
    IsFetchingHorizonData = false;
    OnPropertyChanged(nameof(HasSpkRecords));
    OnPropertyChanged(nameof(HasNoSpkRecords));
    OnPropertyChanged(nameof(CanGoNext));
  }

  [RelayCommand]
  private void ResetRadius()
  {
    CometRadiusKm = _jplRadiusKm;
  }

  [RelayCommand]
  private async Task ValidateTimelineAsync()
  {
    if (!IsEpochRangeValid || SelectedSpkRecord == null)
      return;

    IsValidatingTimeline = true;
    TimelineValidationStatus = "Checking SPK coverage...";

    int naifId = int.TryParse(SelectedSpkRecord.RecordId, out int id) ? id : 0;
    if (naifId == 0)
    {
      TimelineValidationStatus = "Invalid SPK record ID";
      IsValidatingTimeline = false;
      return;
    }

    // 1. Check existing almanac coverage
    bool hasCoverage = _runtimeService.CheckAlmanacCoverage(
      naifId, WizardStartEpoch, WizardEndEpoch);

    if (!hasCoverage)
    {
      // 2. Download SPK from JPL Horizons
      TimelineValidationStatus = "Downloading SPK ephemeris from JPL Horizons...";
      var loadingMsg = _breadcrumbService.ShowLoadingMessage(
        "SPK Download", "Downloading comet ephemeris data...");

      try
      {
        var savePath = _horizonService.GetSpkSavePath(naifId);
        var startStr = WizardStartEpoch.ToString("yyyy-MM-dd");
        var stopStr = WizardEndEpoch.ToString("yyyy-MM-dd");

        var spkPath = await _horizonService.DownloadSpkByIdAsync(
          SelectedComet?.PrimaryDesignation ?? naifId.ToString(),
          SelectedSpkRecord.RecordId,
          savePath,
          startStr,
          stopStr
        );

        if (spkPath != null)
        {
          // 3. Load into almanac
          TimelineValidationStatus = "Loading SPK into almanac...";
          await _runtimeService.LoadAlmanacFileAsync(spkPath);

          // 4. Re-check coverage
          hasCoverage = _runtimeService.CheckAlmanacCoverage(
            naifId, WizardStartEpoch, WizardEndEpoch);
        }
        else
        {
          TimelineValidationStatus = "SPK download failed. Try adjusting the epoch range.";
        }
      }
      catch (Exception ex)
      {
        TimelineValidationStatus = $"Download error: {ex.Message}";
      }
      finally
      {
        _breadcrumbService.RemoveMessage(loadingMsg);
      }
    }

    if (hasCoverage)
    {
      IsTimelineValidated = true;
      TimelineValidationStatus = "SPK coverage verified for Earth + comet";
    }
    else if (string.IsNullOrEmpty(TimelineValidationStatus) || TimelineValidationStatus.StartsWith("Checking"))
    {
      TimelineValidationStatus = "SPK data unavailable for this epoch interval. Try different dates.";
    }

    IsValidatingTimeline = false;
    OnPropertyChanged(nameof(CanGoNext));
    OnPropertyChanged(nameof(CanValidateTimeline));
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

  // ── Helpers ────────────────────────────────────────────────────────────────

  /// <summary>NAIF ID parsed from the selected SPK record. 0 if not applicable.</summary>
  public int SpkNaifId => int.TryParse(SelectedSpkRecord?.RecordId, out int id) ? id : 0;

  /// <summary>Primary designation of the chosen comet (e.g. "1P").</summary>
  public string? CometDesignation => SelectedComet?.PrimaryDesignation;
}
