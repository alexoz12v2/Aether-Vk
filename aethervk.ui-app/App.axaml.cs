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

  // Keep a static reference so the delegate doesn't get garbage collected
  private static AetherVk.Logic.Services.NativeInterop.PanicCallbackDelegate _rustPanicCallback =
    OnRustPanic;

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
      if (
        Application.Current?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop
      )
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
              var errorWindow = new Views.FatalErrorWindow(m.Message);

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
        desktop.MainWindow = new Views.FatalErrorWindow(
          $"The required native library '{libName}' was not found in the executable directory.\n\nThe application cannot run without the core simulation engine."
        );
      }
      else
      {
        AetherVk.Logic.Services.NativeInterop.avkSimulationContext_registerPanicCallback(
          _rustPanicCallback
        );

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

            var inputRegistry =
              Microsoft.Extensions.DependencyInjection.ServiceProviderServiceExtensions.GetRequiredService<AetherVk.Logic.Input.InputRegistry>(
                App.Host!.Services
              );


            // TODO LOAD INPUT BINDIINGS INTO INPUT REGISTRY

            // Attach global router
            var globalRouter = new AetherVk.Input.GlobalInputRouter(mainWindow, inputRegistry);

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
          System.Environment.Exit(0);
        };
      }
    }

    base.OnFrameworkInitializationCompleted();
  }
}
