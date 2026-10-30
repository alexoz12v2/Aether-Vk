using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Reactive.Concurrency;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using AetherVk.Logic.Attributes;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

[GenerateLocalizedStrings(keyPrefix: "Tabs_Settings_", designTitle: "Settings", designIcon: "⚙")]
public partial class SettingsTabViewModel
  : StatefulTabViewModelBase<SettingsSession>,
    ISettingsTabViewModel
{
  private readonly ITranslationService _translationService;
  private readonly CompositeDisposable _disposables = [];
  private readonly INativeRuntimeService _runtimeService;
  private readonly ISchedulerProvider _schedulerProvider;
  private readonly PreferencesService _preferences;
  private readonly TimeScaleService _timeScale;

  /// <summary>
  /// Time scale for dates: 0 = UTC (what a telescope image's DATE-OBS uses, the default), 1 = TDB
  /// (the engine's dynamical time). Saved to preferences.json; the Comet and Timeline boxes, the
  /// overlay label and the FITS facts follow it. The committed range stays TDB parts.
  /// </summary>
  public int TimeScaleIndex
  {
    get => _timeScale.Scale == Services.TimeScale.Tdb ? 1 : 0;
    set
    {
      var scale = value == 1 ? Services.TimeScale.Tdb : Services.TimeScale.Utc;
      if (scale == _timeScale.Scale)
        return;
      _timeScale.Scale = scale;
      OnPropertyChanged();
      OnPropertyChanged(nameof(TimeScaleHelpText));
    }
  }

  /// <summary>The help next to the picker: the two scales, today's offset and the active
  /// leap-second table ("Leap seconds: naif0012.tls, last +37 s on 2017-01-01").</summary>
  public string TimeScaleHelpText => _timeScale.HelpText;

  /// <summary>The diagnostic HUD panels over every viewport (camera coordinates, dust history,
  /// observer line, orbit debug); hidden by default, saved to preferences.json.</summary>
  public bool ShowDiagnosticPanels
  {
    get => _preferences.ShowDiagnosticPanels;
    set
    {
      if (value == _preferences.ShowDiagnosticPanels)
        return;
      _preferences.SetShowDiagnosticPanels(value);
      OnPropertyChanged();
    }
  }

  [ObservableProperty]
  private string _cameraModeName = "Up Zenith";

  [ObservableProperty]
  private bool _hasActiveViewport = false;

  public ObservableCollection<ViewportSettingsViewModel> ActiveViewports { get; } = new();

  /// <summary>Comet Appearance section (display mode, custom mesh wiring and placement).</summary>
  public CometAppearanceViewModel CometAppearance { get; }

  /// <summary>Billboards section (imported textures shown in their own desktop windows).</summary>
  public DesktopBillboardsViewModel Billboards { get; }

  /// <summary>Input bindings section (rebind / add / remove chords, reset; saved to preferences.json).</summary>
  public InputBindingsViewModel InputBindings { get; }

  /// <summary>Particle rendering section (dust volume detail and temporal reuse tolerance; saved to
  /// preferences.json).</summary>
  public ParticleRenderingViewModel ParticleRendering { get; }

  public SettingsTabViewModel(
    ITranslationService translationService,
    ISchedulerProvider schedulerProvider,
    ITabStateService<SettingsSession> sessionService,
    ICameraServiceRegistry cameraServiceRegistry,
    INativeRuntimeService runtimeService,
    AssetLibraryService assetLibrary,
    TimelineService timelineService,
    DesktopBillboardService billboardService,
    IScreenInfoProvider screenInfoProvider,
    Input.InputRegistry inputRegistry,
    PreferencesService preferences,
    IFileDialogService? fileDialogService = null,
    FitsImageService? fitsImageService = null,
    ViewModeBarViewModel? modeBar = null,
    TimeScaleService? timeScale = null
  )
    : base("Settings", sessionService)
  {
    _translationService = translationService;
    _runtimeService = runtimeService;
    _schedulerProvider = schedulerProvider;
    _preferences = preferences;
    _timeScale = timeScale ?? TimeScaleService.TdbOnly();
    // a switch from elsewhere (the boxes' "Use TDB for dates" action): the picker follows
    _timeScale.ScaleChanged += OnTimeScaleChanged;
    _disposables.Add(System.Reactive.Disposables.Disposable.Create(() => _timeScale.ScaleChanged -= OnTimeScaleChanged));
    CometAppearance = new CometAppearanceViewModel(
      runtimeService,
      assetLibrary,
      timelineService,
      translationService,
      schedulerProvider);
    InputBindings = new InputBindingsViewModel(inputRegistry, preferences);
    ParticleRendering = new ParticleRenderingViewModel(runtimeService, preferences);
    Billboards = new DesktopBillboardsViewModel(
      billboardService,
      assetLibrary,
      screenInfoProvider,
      schedulerProvider,
      fitsImageService,
      timelineService.SeekAsync,
      timelineService.RequestModelRange,
      translationService);
    // the main window's Model button runs the FITS comparison through these rows
    modeBar?.AttachBillboards(Billboards).AddDisposableTo(_disposables);

    Icon = "⚙";

    SubscribeToStrings(schedulerProvider);

    cameraServiceRegistry.ViewportCreated
      .ObserveOn(schedulerProvider.Background)
      .Subscribe(cameraId =>
      {
        var cameraService = cameraServiceRegistry.Get(cameraId);
        if (cameraService != null)
        {
          schedulerProvider.MainThread.Schedule(() =>
          {
            int nextIndex = ActiveViewports.Count;
            var vm = new ViewportSettingsViewModel(
              cameraId, nextIndex, _runtimeService, _schedulerProvider, cameraService);
            ActiveViewports.Add(vm);
            HasActiveViewport = true;
            Billboards.AttachViewport(new CameraBillboardProbe(cameraService));

            // Simple fallback to keep the global mode display somewhat functional
            if (ActiveViewports.Count == 1)
            {
              cameraService.CameraModeChanged
                .ObserveOn(schedulerProvider.MainThread)
                .Subscribe(mode => CameraModeName = mode.ToString())
                .AddDisposableTo(_disposables);
            }
          });
        }
      })
      .AddDisposableTo(_disposables);

    cameraServiceRegistry.ViewportDestroyed
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(cameraId =>
      {
        var vm = ActiveViewports.FirstOrDefault(v => v.CameraId == cameraId);
        if (vm != null)
        {
          vm.Dispose();
          ActiveViewports.Remove(vm);
        }
        HasActiveViewport = ActiveViewports.Count > 0;
      })
      .AddDisposableTo(_disposables);
  }

  private void OnTimeScaleChanged(TimeScale _)
  {
    OnPropertyChanged(nameof(TimeScaleIndex));
    OnPropertyChanged(nameof(TimeScaleHelpText));
  }

  public override void Dispose()
  {
    CometAppearance.Dispose();
    _disposables.Dispose();
    base.Dispose();
  }

  private void SubscribeToStrings(ISchedulerProvider schedulerProvider)
  {
    RefreshAllStrings();
    _translationService
      .CultureChanged.Skip(1)
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(_ => RefreshAllStrings())
      .AddDisposableTo(_disposables);
  }

  /// <summary>The generated strings, then the ones a section shows as picker items.</summary>
  private void RefreshAllStrings()
  {
    RefreshStrings();
    ParticleRendering.ReuseToleranceOffLabel = StrReuseToleranceOff;
  }
}
