using System;
using System.Collections.ObjectModel;
using AetherVk.Logic.Models;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

public partial class AssetBrowserViewModel : TabItemViewModel
{
  private readonly IServiceProvider _serviceProvider;

  public MainWindowViewModel MainViewModel =>
    (MainWindowViewModel)_serviceProvider.GetService(typeof(MainWindowViewModel))!;

  public AssetBrowserViewModel(IServiceProvider serviceProvider)
    : base("Asset Browser")
  {
    _serviceProvider = serviceProvider;
    Icon = "📦";
  }
}
