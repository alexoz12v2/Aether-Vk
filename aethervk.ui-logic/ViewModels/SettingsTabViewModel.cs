using System;
using System.Collections.ObjectModel;
using System.Linq;
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
  private readonly CameraService _cameraService;
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
    CameraService cameraService,
    INativeRuntimeService runtimeService,
    IViewportRegistry viewportRegistry
  )
    : base("Settings", sessionService)
  {
    _translationService = translationService;
    _cameraService = cameraService;
    _runtimeService = runtimeService;
    _schedulerProvider = schedulerProvider;

    Icon = "⚙";

    SubscribeToStrings(schedulerProvider);

    _cameraService
      .CameraModeChanged.ObserveOn(schedulerProvider.MainThread)
      .Subscribe(mode => CameraModeName = mode.ToString())
      .AddDisposableTo(_disposables);

    viewportRegistry
      .ActiveViewports.ObserveOn(schedulerProvider.MainThread)
      .Subscribe(list =>
      {
        HasActiveViewport = list.Length > 0;

        // Remove viewports that no longer exist
        var toRemove = ActiveViewports
          .Where(vm => !list.Any(e => e.CameraId == vm.CameraId))
          .ToList();
        foreach (var vm in toRemove)
        {
          vm.Dispose();
          ActiveViewports.Remove(vm);
        }

        // Add new viewports
        for (int i = 0; i < list.Length; i++)
        {
          var entry = list[i];
          if (!ActiveViewports.Any(vm => vm.CameraId == entry.CameraId))
          {
            ActiveViewports.Add(
              new ViewportSettingsViewModel(entry.CameraId, i, _runtimeService, _schedulerProvider)
            );
          }
        }
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
