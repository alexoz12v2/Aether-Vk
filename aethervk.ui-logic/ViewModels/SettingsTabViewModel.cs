using System;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using AetherVk.Logic.Attributes;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;

namespace AetherVk.Logic.ViewModels;

// TODO tab header should also be localized — accept key once translation service
// is wired into StatefulTabViewModelBase.
[GenerateLocalizedStrings(
  keyPrefix: "Tabs_Settings_",
  designTitle: "Settings",
  designIcon: "⚙")]
public partial class SettingsTabViewModel
  : StatefulTabViewModelBase<SettingsSession>,
    ISettingsTabViewModel
{
  private readonly ITranslationService _translationService;
  private readonly CompositeDisposable _disposables = [];

  public SettingsTabViewModel(
    ITranslationService translationService,
    ISchedulerProvider schedulerProvider,
    ITabStateService<SettingsSession> sessionService)
    : base("Settings", sessionService)
  {
    _translationService = translationService;

    Icon = "⚙"; // gear — U+2699

    // RefreshStrings() is generated; SubscribeToStrings wires the culture-change
    // subscription so the generated refresh is also called on locale switch.
    SubscribeToStrings(schedulerProvider);
  }

  private void SubscribeToStrings(ISchedulerProvider schedulerProvider)
  {
    RefreshStrings(); // initial population (generated method)
    _translationService.CultureChanged
      .Skip(1) // skip the replay of the current culture on subscribe
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(_ => RefreshStrings())
      .AddDisposableTo(_disposables);
  }
}
