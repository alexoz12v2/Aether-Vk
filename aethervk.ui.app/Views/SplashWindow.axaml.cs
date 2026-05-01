using System;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Threading;

namespace AetherVk.Views;

public partial class SplashWindow : Window
{
  public SplashWindow()
  {
    InitializeComponent();
    Loaded += OnLoaded;
  }

  private async void OnLoaded(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
  {
    bool success = false;
    string errorMessage = "Unknown error";

    var runtimeService =
      ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;

    if (runtimeService == null)
    {
      ShowError(
        "Critical Failure: NativeRuntimeService not registered in dependency injection container."
      );
      return;
    }

    try
    {
      await Task.Run(() =>
      {
        // Init native simulation engine
        runtimeService.InitializeSimulationContext("Vulkan", null, false);

        // Create default scene natively
        runtimeService.CreateScene(false);
      });

      success = true;
    }
    catch (Exception ex)
    {
      errorMessage = ex.Message;
    }

    if (success)
    {
      Dispatcher.UIThread.Post(() =>
      {
        if (
          Application.Current?.ApplicationLifetime
          is IClassicDesktopStyleApplicationLifetime desktop
        )
        {
          var mainWindowViewModel = new MainWindowViewModel();
          var mainWindow = new MainWindow { DataContext = mainWindowViewModel };

          // Listen for theme changes in the ViewModel
          mainWindowViewModel.PropertyChanged += (vmSender, vmArgs) =>
          {
            if (vmArgs.PropertyName == nameof(MainWindowViewModel.CurrentTheme))
            {
              if (vmSender is MainWindowViewModel vm)
              {
                Application.Current.RequestedThemeVariant = vm.CurrentTheme switch
                {
                  AppTheme.Light => Avalonia.Styling.ThemeVariant.Light,
                  AppTheme.Dark => Avalonia.Styling.ThemeVariant.Dark,
                  _ => Avalonia.Styling.ThemeVariant.Default,
                };
              }
            }
          };

          desktop.MainWindow = mainWindow;
          mainWindow.Show();
          this.Close();
        }
      });
    }
    else
    {
      Dispatcher.UIThread.Post(() => ShowError(errorMessage));
    }
  }

  private void ShowError(string message)
  {
    var errorWindow = new Window
    {
      Title = "Critical Failure",
      Width = 600,
      Height = 200,
      WindowStartupLocation = Avalonia.Controls.WindowStartupLocation.CenterScreen,
      Content = new StackPanel
      {
        Children =
        {
          new TextBlock
          {
            Text =
              $"CRITICAL ERROR:\n{message}\n\nThe application cannot run without the core simulation engine.",
            Foreground = Avalonia.Media.Brushes.Red,
            FontWeight = Avalonia.Media.FontWeight.Bold,
            FontSize = 16,
            Margin = new Avalonia.Thickness(20),
            TextWrapping = Avalonia.Media.TextWrapping.Wrap,
          },
          new Button
          {
            Content = "Ok",
            HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center,
            Margin = new Avalonia.Thickness(10),
          },
        },
      },
    };

    var btn = (Button)((StackPanel)errorWindow.Content).Children[1];
    btn.Click += (s, e) =>
    {
      errorWindow.Close();
      Environment.Exit(1);
    };

    if (Application.Current?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
    {
      desktop.MainWindow = errorWindow;
    }

    errorWindow.Show();
    this.Close();
  }
}
