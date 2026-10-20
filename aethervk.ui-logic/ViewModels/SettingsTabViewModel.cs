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
  private readonly IFileDialogService? _fileDialogService;

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
    IFileDialogService? fileDialogService = null
  )
    : base("Settings", sessionService)
  {
    _translationService = translationService;
    _fileDialogService = fileDialogService;
    _runtimeService = runtimeService;
    _schedulerProvider = schedulerProvider;
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
      schedulerProvider);

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
              cameraId, nextIndex, _runtimeService, _schedulerProvider, cameraService,
              _fileDialogService, _translationService);
            ActiveViewports.Add(vm);
            HasActiveViewport = true;

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
