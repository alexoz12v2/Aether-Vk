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

  [ObservableProperty]
  private string _cameraModeName = "Up Zenith";

  [ObservableProperty]
  private bool _hasActiveViewport = false;

  public ObservableCollection<ViewportSettingsViewModel> ActiveViewports { get; } = new();

  public SettingsTabViewModel(
    ITranslationService translationService,
    ISchedulerProvider schedulerProvider,
    ITabStateService<SettingsSession> sessionService,
    ICameraServiceRegistry cameraServiceRegistry,
    INativeRuntimeService runtimeService
  )
    : base("Settings", sessionService)
  {
    _translationService = translationService;
    _runtimeService = runtimeService;
    _schedulerProvider = schedulerProvider;

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
            var vm = new ViewportSettingsViewModel(cameraId, nextIndex, _runtimeService, _schedulerProvider, cameraService);
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

  private void SubscribeToStrings(ISchedulerProvider schedulerProvider)
  {
    RefreshStrings();
    _translationService
      .CultureChanged.Skip(1)
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(_ => RefreshStrings())
      .AddDisposableTo(_disposables);
  }
}
