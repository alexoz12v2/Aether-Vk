using System;
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
using Microsoft.Extensions.DependencyInjection;

namespace AetherVk;

public partial class App : Application
{
  public static IHost? Host { get; set; }

  private static void OnRustPanic(IntPtr messagePtr, nuint length)
  {
    string errorMsg = "Unknown Rust Panic";
    if (messagePtr != IntPtr.Zero)
    {
      errorMsg =
        System.Runtime.InteropServices.Marshal.PtrToStringAnsi(messagePtr, (int)length) ?? errorMsg;
    }

    Avalonia.Threading.Dispatcher.UIThread.Post(() =>
    {
      if (Current?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
      {
        var oldMain = desktop.MainWindow;
        var errorWindow = new Views.FatalErrorWindow(
          $"The Rust Core Engine panicked and cannot recover.\n\nDetails:\n{errorMsg}"
        );
        desktop.MainWindow = errorWindow;
        errorWindow.Show();
        oldMain?.Close();
      }
    });
  }

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
#if DEBUG
      if (desktop.Args?.Contains("--force-fatal-error") == true)
      {
        desktop.MainWindow = new Views.FatalErrorWindow(
          "This is a simulated fatal error for debugging the graphics of the fatal error window."
        );
        base.OnFrameworkInitializationCompleted();
        return;
      }
#endif

      WeakReferenceMessenger.Default.Register<App, Logic.Messages.CriticalErrorMessage>(this,
        (r, m) =>
        {
          Avalonia.Threading.Dispatcher.UIThread.Post(() =>
          {
            if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime dt)
            {
              var errorWindow = new Views.FatalErrorWindow(m.Message);

              var oldMain = dt.MainWindow;
              dt.MainWindow = errorWindow;
              errorWindow.Show();
              oldMain?.Close();
            }
          });
        }
      );

      WeakReferenceMessenger.Default.Register<App, Logic.Messages.CopyToClipboardMessage>(this,
        async (r, m) =>
        {
          var cb = TopLevel.GetTopLevel(desktop.MainWindow)?.Clipboard;
          if (cb != null)
          {
            await cb.SetTextAsync(m.Text);
          }
        }
      );

      string libName = OperatingSystem.IsWindows() ? "aethervk_core.dll" : "libaethervk_core.so";

      // Fallback check in case the user runs the app from the CLI without correct working directory
      string libPath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, libName);

      if (!System.IO.File.Exists(libPath) && !System.IO.File.Exists(libName))
      {
        desktop.MainWindow = new Views.FatalErrorWindow(
          $"The required native library '{libName}' was not found in the executable directory.\n\nThe application cannot run without the core simulation engine."
        );
      }
      else
      {
        var runtimeService = ServiceProviderServiceExtensions.GetRequiredService<INativeRuntimeService>(Host!.Services);
        runtimeService.RegisterPanicCallback(OnRustPanic);

        var splashViewModel = new SplashViewModel(runtimeService);
        var splashWindow = new Views.SplashWindow { DataContext = splashViewModel };

        splashViewModel.OnInitializationCompleted += () =>
        {
          Avalonia.Threading.Dispatcher.UIThread.Post(() =>
          {
            var inputRegistry = ServiceProviderServiceExtensions.GetRequiredService<Logic.Input.InputRegistry>(Host!.Services);
            // TODO LOAD INPUT BINDIINGS INTO INPUT REGISTRY
            // Configure the singleton before instantiating main window view model

            var mainWindowViewModel = ServiceProviderServiceExtensions.GetRequiredService<MainWindowViewModel>(Host!.Services);
            var mainWindow = new MainWindow { DataContext = mainWindowViewModel };

            // Listen for theme changes in the ViewModel
            mainWindowViewModel.PropertyChanged += (vmSender, vmArgs) =>
            {
              if (vmArgs.PropertyName == nameof(MainWindowViewModel.CurrentTheme))
              {
                if (vmSender is MainWindowViewModel vm)
                {
                  Current!.RequestedThemeVariant = vm.CurrentTheme switch
                  {
                    AppTheme.Light => ThemeVariant.Light,
                    AppTheme.Dark => ThemeVariant.Dark,
                    _ => ThemeVariant.Default,
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
            var errorWindow = new Views.FatalErrorWindow(
              $"{errorMessage}\n\nThe application cannot run without the core simulation engine."
            );
            desktop.MainWindow = errorWindow;
            errorWindow.Show();
            splashWindow.Close();
          });
        };

        desktop.MainWindow = splashWindow;
        _ = splashViewModel.InitializeAsync();

        desktop.Exit += (sender, args) =>
        {
          // All native services which are disposable should be disposed of here
          // TODO ensure all dependencies which use the runtime service are cleaned up with a shutdown message?
          runtimeService.Dispose();
          Environment.Exit(0);
        };
      }
    }

    base.OnFrameworkInitializationCompleted();
  }
}
