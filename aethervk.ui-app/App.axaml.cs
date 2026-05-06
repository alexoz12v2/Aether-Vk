using System;
using System.ComponentModel;
using System.Linq;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Data.Core.Plugins;
using Avalonia.Markup.Xaml;
using Avalonia.Styling;
using CommunityToolkit.Mvvm.Messaging;
using Microsoft.Extensions.Hosting;

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
      CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Register<
        App,
        AetherVk.Logic.Messages.CriticalErrorMessage
      >(
        this,
        (r, m) =>
        {
          Avalonia.Threading.Dispatcher.UIThread.Post(() =>
          {
            if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime dt)
            {
              var errorWindow = new Avalonia.Controls.Window
              {
                Title = "Critical Failure",
                Width = 600,
                Height = 200,
                WindowStartupLocation = Avalonia.Controls.WindowStartupLocation.CenterScreen,
                Content = new Avalonia.Controls.TextBlock
                {
                  Text = m.Message,
                  Foreground = Avalonia.Media.Brushes.Red,
                  FontWeight = Avalonia.Media.FontWeight.Bold,
                  FontSize = 16,
                  Margin = new Avalonia.Thickness(20),
                  TextWrapping = Avalonia.Media.TextWrapping.Wrap,
                },
              };

              var oldMain = dt.MainWindow;
              dt.MainWindow = errorWindow;
              errorWindow.Show();
              oldMain?.Close();
            }
          });
        }
      );

      CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Register<
        App,
        AetherVk.Logic.Messages.CopyToClipboardMessage
      >(
        this,
        async (r, m) =>
        {
          var cb = TopLevel.GetTopLevel(desktop.MainWindow)?.Clipboard;
          if (cb != null)
          {
            await cb.SetTextAsync(m.Text);
          }
        }
      );

      string libExtension =
        System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
          System.Runtime.InteropServices.OSPlatform.Windows
        )
          ? ".dll"
        : System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
          System.Runtime.InteropServices.OSPlatform.OSX
        )
          ? ".dylib"
        : ".so";
      string libPrefix = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
        System.Runtime.InteropServices.OSPlatform.Windows
      )
        ? ""
        : "lib";
      string libName = $"{libPrefix}aethervk_core_cdylib{libExtension}";

      // Fallback check in case the user runs the app from the CLI without correct working directory
      string libPath = System.IO.Path.Combine(
        System.AppDomain.CurrentDomain.BaseDirectory,
        libName
      );

      if (!System.IO.File.Exists(libPath) && !System.IO.File.Exists(libName))
      {
        desktop.MainWindow = new Avalonia.Controls.Window
        {
          Title = "Critical Failure",
          Width = 600,
          Height = 200,
          WindowStartupLocation = Avalonia.Controls.WindowStartupLocation.CenterScreen,
          Content = new Avalonia.Controls.TextBlock
          {
            Text =
              $"CRITICAL ERROR:\nThe required native library '{libName}' was not found in the executable directory.\n\nThe application cannot run without the core simulation engine.",
            Foreground = Avalonia.Media.Brushes.Red,
            FontWeight = Avalonia.Media.FontWeight.Bold,
            FontSize = 16,
            Margin = new Avalonia.Thickness(20),
            TextWrapping = Avalonia.Media.TextWrapping.Wrap,
          },
        };
      }
      else
      {
        var runtimeService =
          Microsoft.Extensions.DependencyInjection.ServiceProviderServiceExtensions.GetRequiredService<NativeRuntimeService>(
            App.Host!.Services
          );
        var splashViewModel = new SplashViewModel(runtimeService);
        var splashWindow = new Views.SplashWindow { DataContext = splashViewModel };

        splashViewModel.OnInitializationCompleted += () =>
        {
          Avalonia.Threading.Dispatcher.UIThread.Post(() =>
          {
            var mainWindowViewModel =
              Microsoft.Extensions.DependencyInjection.ServiceProviderServiceExtensions.GetRequiredService<MainWindowViewModel>(
                App.Host!.Services
              );
            var mainWindow = new MainWindow { DataContext = mainWindowViewModel };

            // Listen for theme changes in the ViewModel
            mainWindowViewModel.PropertyChanged += (vmSender, vmArgs) =>
            {
              if (vmArgs.PropertyName == nameof(MainWindowViewModel.CurrentTheme))
              {
                if (vmSender is MainWindowViewModel vm)
                {
                  Application.Current!.RequestedThemeVariant = vm.CurrentTheme switch
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
            splashWindow.Close();
          });
        };

        splashViewModel.OnInitializationFailed += (errorMessage) =>
        {
          Avalonia.Threading.Dispatcher.UIThread.Post(() =>
          {
            var errorWindow = new Avalonia.Controls.Window
            {
              Title = "Critical Failure",
              Width = 600,
              Height = 200,
              WindowStartupLocation = Avalonia.Controls.WindowStartupLocation.CenterScreen,
              Content = new Avalonia.Controls.TextBlock
              {
                Text =
                  $"CRITICAL ERROR:\n{errorMessage}\n\nThe application cannot run without the core simulation engine.",
                Foreground = Avalonia.Media.Brushes.Red,
                FontWeight = Avalonia.Media.FontWeight.Bold,
                FontSize = 16,
                Margin = new Avalonia.Thickness(20),
                TextWrapping = Avalonia.Media.TextWrapping.Wrap,
              },
            };
            desktop.MainWindow = errorWindow;
            errorWindow.Show();
            splashWindow.Close();
          });
        };

        desktop.MainWindow = splashWindow;
        _ = splashViewModel.InitializeAsync();

        desktop.Exit += (sender, args) =>
        {
          runtimeService.Dispose();
          System.Environment.Exit(0);
        };
      }
    }

    base.OnFrameworkInitializationCompleted();
  }
}
