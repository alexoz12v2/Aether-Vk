using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using AetherVk.Logic.Attributes;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

[GenerateLocalizedStrings(
  keyPrefix:    "Tabs_Imports_",
  designTitle:  "Imports",
  designIcon:   "⬇")]
public partial class ImportsTabViewModel : StatefulTabViewModelBase<ImportsSession>, IImportsTabViewModel
{
  private readonly ITranslationService _translationService;
  private readonly ILocalStorageService _localStorageService;
  private readonly CompositeDisposable _disposables = [];

  [ObservableProperty]
  private ObservableCollection<string> _sessionFolders = [];

  public ImportsTabViewModel(
    ITranslationService translationService,
    ISchedulerProvider schedulerProvider,
    ITabStateService<ImportsSession> sessionService,
    ILocalStorageService localStorageService)
    : base("Imports", sessionService)
  {
    _translationService = translationService;
    _localStorageService = localStorageService;
    Icon = "⬇"; // down arrow / import — U+2B07
    SubscribeToStrings(schedulerProvider);

    var sessionsDir = Path.GetDirectoryName(_localStorageService.SessionDirectory);
    if (sessionsDir is not null && Directory.Exists(sessionsDir))
    {
      var dirs = Directory.GetDirectories(sessionsDir);
      SessionFolders = new ObservableCollection<string>(dirs);
    }
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

public partial interface IImportsTabViewModel
{
  ObservableCollection<string> SessionFolders { get; set; }
}

public partial class DesignImportsTabViewModel
{
  public ObservableCollection<string> SessionFolders { get; set; } = new([
    "/mock/session/folder_1",
    "/mock/session/folder_2"
  ]);
}
