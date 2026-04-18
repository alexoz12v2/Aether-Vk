using System.Linq;
using AetherVk.Logic.ViewModels;
using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Data.Core.Plugins;
using Avalonia.Markup.Xaml;
using Avalonia.Styling;
using Microsoft.Extensions.Hosting;
using System;
using System.ComponentModel;

namespace AetherVk;

public partial class App : Application
{
  public static IHost? Host { get; set; }

  public override void Initialize()
  {
    AvaloniaXamlLoader.Load(this);
  }

  public override void OnFrameworkInitializationCompleted()
  {
    // CommunityToolkit has its own data validation. we don't need data validation from Avalonia Too
    var dataValidationPluginsToRemove = BindingPlugins
      .DataValidators.OfType<DataAnnotationsValidationPlugin>()
      .ToArray();
    foreach (var plugin in dataValidationPluginsToRemove)
    {
      BindingPlugins.DataValidators.Remove(plugin);
    }

    if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
    {
      var mainWindowViewModel = new MainWindowViewModel();
      desktop.MainWindow = new MainWindow { DataContext = mainWindowViewModel };

      // Listen for theme changes in the ViewModel
      mainWindowViewModel.PropertyChanged += (sender, e) =>
      {
          if (e.PropertyName == nameof(MainWindowViewModel.CurrentTheme))
          {
              if (sender is MainWindowViewModel vm)
              {
                  RequestedThemeVariant = vm.CurrentTheme switch
                  {
                      AppTheme.Light => ThemeVariant.Light,
                      AppTheme.Dark => ThemeVariant.Dark,
                      _ => ThemeVariant.Default,
                  };
              }
          }
      };
    }

    base.OnFrameworkInitializationCompleted();
  }
}
