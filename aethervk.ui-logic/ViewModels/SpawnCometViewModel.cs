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

  // ── SBDB data (canonical NAIF SPKID) ────────────────────────────────────────

  /// <summary>Cached SBDB data for the currently selected comet. Null until fetched.</summary>
  private SmallBodyDataComponent? _sbdbData;

  /// <summary>Designation for which _sbdbData was fetched. Prevents re-fetch on back/forward.</summary>
  private string? _sbdbCachedDesignation;

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
  private double _massKg = 1.0;

  // IAU Rotational Model parameters (from PhysicalMeshComponent)
  [ObservableProperty]
  private double _poleRaDeg = 90.0;

  [ObservableProperty]
  private double _poleDecDeg = 90.0 - IauRotationMath.ObliquityDeg;

  [ObservableProperty]
  private double _primeMeridianDeg = 180.0;

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
  [NotifyPropertyChangedFor(nameof(EpochSummary))]
  private DateTimeOffset _wizardStartEpoch;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(IsEpochRangeValid))]
  [NotifyPropertyChangedFor(nameof(CanValidateTimeline))]
  [NotifyPropertyChangedFor(nameof(EpochSummary))]
  private DateTimeOffset _wizardEndEpoch;

  /// <summary>True when start is strictly before end.</summary>
  public bool IsEpochRangeValid => WizardStartEpoch < WizardEndEpoch;

  /// <summary>Can click Validate only when range is valid and not already validated.</summary>
  public bool CanValidateTimeline => IsEpochRangeValid && !IsTimelineValidated && !IsValidatingTimeline;

  /// <summary>Human-readable epoch range for Step 4 display.</summary>
  public string EpochSummary => $"Epoch: {WizardStartEpoch:yyyy-MM-dd} → {WizardEndEpoch:yyyy-MM-dd}";

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
    if (SelectedComet == null || !IsEpochRangeValid)
      return;
    IsFetchingHorizonData = true;
    SelectedSpkRecord = null;
    IsTimelineValidated = false;
    TimelineValidationStatus = "";
    var start = WizardStartEpoch.ToString("yyyy-MM-dd");
    var stop = WizardEndEpoch.ToString("yyyy-MM-dd");
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
    TimelineValidationStatus = "";

    // Use the canonical NAIF SPKID from SBDB if available, otherwise fall back
    // to the Horizons record number (the probe will discover the real ID).
    int naifId = _sbdbData?.SpkId ?? 0;
    if (naifId == 0)
      naifId = int.TryParse(SelectedSpkRecord.RecordId, out int id) ? id : 0;
    if (naifId == 0)
    {
      TimelineValidationStatus = "Invalid SPK record ID";
      IsValidatingTimeline = false;
      return;
    }

    // 1. Check if existing almanac already covers this range
    bool existingCoverage = _runtimeService.CheckAlmanacCoverage(
      naifId, WizardStartEpoch, WizardEndEpoch);

    string? downloadedSpkPath = null;

    if (existingCoverage)
    {
      // Already covered — skip download, go straight to final load confirmation
      IsTimelineValidated = true;
      TimelineValidationStatus = "✓ SPK coverage already loaded for this epoch range";
      IsValidatingTimeline = false;
      OnPropertyChanged(nameof(CanGoNext));
      OnPropertyChanged(nameof(CanValidateTimeline));
      return;
    }

    // 2. Download SPK from JPL Horizons
    var downloadMsg = _breadcrumbService.ShowLoadingMessage(
      "SPK Download", "Downloading comet ephemeris from JPL Horizons…");

    try
    {
      var savePath = _horizonService.GetSpkSavePath(naifId);
      var startStr = WizardStartEpoch.ToString("yyyy-MM-dd");
      var stopStr = WizardEndEpoch.ToString("yyyy-MM-dd");

      downloadedSpkPath = await _horizonService.DownloadSpkByIdAsync(
        SelectedComet?.PrimaryDesignation ?? naifId.ToString(),
        SelectedSpkRecord.RecordId,
        savePath,
        startStr,
        stopStr
      );
    }
    catch (Exception ex)
    {
      _breadcrumbService.RemoveMessage(downloadMsg);
      _breadcrumbService.ShowErrorMessage("SPK Error", $"Download failed: {ex.Message}");
      TimelineValidationStatus = $"Download error: {ex.Message}";
      IsValidatingTimeline = false;
      return;
    }

    _breadcrumbService.RemoveMessage(downloadMsg);

    if (string.IsNullOrEmpty(downloadedSpkPath))
    {
      _breadcrumbService.ShowErrorMessage("SPK Error", "SPK download failed. Try adjusting the epoch range.");
      TimelineValidationStatus = "SPK download failed. Try adjusting the epoch range.";
      IsValidatingTimeline = false;
      return;
    }

    // 3. Probe SPK in temporary almanac (synchronous, no locks on simulation state)
    var probeMsg = _breadcrumbService.ShowLoadingMessage(
      "SPK Probe", "Verifying ephemeris coverage at epoch boundaries…");

    bool probeOk;
    DateTimeOffset? domainStart = null, domainEnd = null;
    int discoveredNaifId = 0;
    try
    {
      // Run on thread pool to avoid blocking UI — the native call mmaps the file
      var result = await Task.Run(() =>
        _runtimeService.ProbeSpkFile(downloadedSpkPath, naifId, WizardStartEpoch, WizardEndEpoch)
      );
      probeOk = result.covers;
      domainStart = result.domainStart;
      domainEnd = result.domainEnd;
      discoveredNaifId = result.discoveredNaifId;

      // If SBDB wasn't available, store the discovered NAIF ID so downstream
      // code (SpkNaifId, AddAlmanacPlanet) uses the correct physical ID.
      if (discoveredNaifId != 0 && (_sbdbData == null || _sbdbData.SpkId == 0))
      {
        _sbdbData ??= new SmallBodyDataComponent();
        _sbdbData.SpkId = discoveredNaifId;
        _sbdbData.Designation = SelectedComet?.PrimaryDesignation ?? string.Empty;
      }
    }
    catch
    {
      probeOk = false;
    }

    _breadcrumbService.RemoveMessage(probeMsg);

    if (!probeOk)
    {
      // 4. Probe failed — delete the downloaded SPK and show error with actual domain
      try { System.IO.File.Delete(downloadedSpkPath); } catch { /* best effort */ }

      string domainInfo = (domainStart.HasValue && domainEnd.HasValue)
        ? $"SPK contains data from {domainStart.Value:yyyy-MM-dd} to {domainEnd.Value:yyyy-MM-dd}."
        : $"Could not find NAIF ID {naifId} in the downloaded SPK file.";

      _breadcrumbService.ShowErrorMessage(
        "SPK Coverage Error",
        $"The downloaded SPK does not cover {WizardStartEpoch:yyyy-MM-dd} to {WizardEndEpoch:yyyy-MM-dd}. " +
        domainInfo + " Adjust dates and try again.");
      TimelineValidationStatus =
        $"Ephemeris probe failed — {domainInfo}";
      IsValidatingTimeline = false;
      return;
    }

    // 5. Probe succeeded — load SPK into the real simulation almanac
    var loadMsg = _breadcrumbService.ShowLoadingMessage(
      "Loading SPK", "Loading SPK into simulation almanac…");

    try
    {
      await _runtimeService.LoadAlmanacFileAsync(downloadedSpkPath);
    }
    catch (Exception ex)
    {
      _breadcrumbService.RemoveMessage(loadMsg);
      _breadcrumbService.ShowErrorMessage("Almanac Error", $"Failed to load SPK: {ex.Message}");
      TimelineValidationStatus = $"Almanac load error: {ex.Message}";
      IsValidatingTimeline = false;
      return;
    }

    _breadcrumbService.RemoveMessage(loadMsg);

    // 6. Success!
    IsTimelineValidated = true;
    TimelineValidationStatus = "✓ SPK coverage verified — ephemeris confirmed at start and end epochs";

    IsValidatingTimeline = false;
    OnPropertyChanged(nameof(CanGoNext));
    OnPropertyChanged(nameof(CanValidateTimeline));
  }

  [RelayCommand]
  private async Task NextStepAsync()
  {
    if (!CanGoNext) return;

    // Fetch SBDB data when transitioning from Step 3 → Step 4
    if (CurrentStep == 3 && SelectedComet != null)
    {
      var des = SelectedComet.PrimaryDesignation;
      if (_sbdbData == null || _sbdbCachedDesignation != des)
      {
        var loadMsg = _breadcrumbService.ShowLoadingMessage(
          "SBDB", "Resolving canonical NAIF SPKID…");
        try
        {
          _sbdbData = await _horizonService.FetchSmallBodyDataAsync(des);
          _sbdbCachedDesignation = des;
        }
        catch { /* non-fatal — probe fallback will discover the ID */ }
        finally { _breadcrumbService.RemoveMessage(loadMsg); }
      }
    }

    CurrentStep++;
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

  /// <summary>
  /// Canonical NAIF SPKID from SBDB, falling back to the Horizons record number.
  /// This is the ID actually stored in the SPK file (e.g. 1000012 for 67P).
  /// </summary>
  public int SpkNaifId => _sbdbData?.SpkId ?? (int.TryParse(SelectedSpkRecord?.RecordId, out int id) ? id : 0);

  /// <summary>SBDB metadata for the selected comet (null if not fetched).</summary>
  public SmallBodyDataComponent? SmallBodyData => _sbdbData;

  /// <summary>Primary designation of the chosen comet (e.g. "1P").</summary>
  public string? CometDesignation => SelectedComet?.PrimaryDesignation;
}
