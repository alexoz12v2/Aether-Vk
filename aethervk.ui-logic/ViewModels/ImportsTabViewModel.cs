using System;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using AetherVk.Logic.Attributes;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;

namespace AetherVk.Logic.ViewModels;

[GenerateLocalizedStrings(
  keyPrefix:    "Tabs_Imports_",
  designTitle:  "Imports",
  designIcon:   "⬇")]
public partial class ImportsTabViewModel : StatefulTabViewModelBase<ImportsSession>, IImportsTabViewModel
{
  private readonly ITranslationService _translationService;
  private readonly CompositeDisposable _disposables = [];

  public ImportsTabViewModel(
    ITranslationService translationService,
    ISchedulerProvider schedulerProvider,
    ITabStateService<ImportsSession> sessionService)
    : base("Imports", sessionService)
  {
    _translationService = translationService;
    Icon = "⬇"; // down arrow / import — U+2B07
    SubscribeToStrings(schedulerProvider);
  }

  private void SubscribeToStrings(ISchedulerProvider schedulerProvider)
  {
    RefreshStrings();
    _translationService.CultureChanged
      .Skip(1)
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(_ => RefreshStrings())
      .AddDisposableTo(_disposables);
  }
}
