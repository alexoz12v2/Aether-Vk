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
  private static AetherVk.Logic.Services.NativeInterop.PanicCallbackDelegate _rustPanicCallback = OnRustPanic;

  private static void OnRustPanic(IntPtr messagePtr, nuint length)
  {
      string errorMsg = "Unknown Rust Panic";
      if (messagePtr != IntPtr.Zero)
      {
          errorMsg = System.Runtime.InteropServices.Marshal.PtrToStringAnsi(messagePtr, (int)length) ?? errorMsg;
      }
      
      Avalonia.Threading.Dispatcher.UIThread.Post(() =>
      {
          if (Application.Current?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
          {
              var oldMain = desktop.MainWindow;
              var errorWindow = new Views.FatalErrorWindow($"The Rust Core Engine panicked and cannot recover.\n\nDetails:\n{errorMsg}");
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
        AetherVk.Logic.Services.NativeInterop.avkSimulationContext_registerPanicCallback(_rustPanicCallback);

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

            inputRegistry.Register(
              "PropertiesPanel",
              new AetherVk.Logic.Input.InputChord(Key: "S", Shift: true),
              new AetherVk.Logic.Input.AppAction(
                "ui.expand_all",
                "Toggle Expanders",
                "Expands all sections"
              )
            );
            inputRegistry.Register(
              "PropertiesPanel",
              new AetherVk.Logic.Input.InputChord(Key: "G", Shift: true),
              new AetherVk.Logic.Input.AppAction(
                "ui.show_flyout",
                "Quick Menu",
                "Opens context menu"
              )
            );
            inputRegistry.Register(
              "PropertiesPanel",
              new AetherVk.Logic.Input.InputChord(Key: "D1"),
              new AetherVk.Logic.Input.AppAction("ui.add_cube", "Add Cube")
            );
            inputRegistry.Register(
              "PropertiesPanel",
              new AetherVk.Logic.Input.InputChord(Key: "Escape"),
              new AetherVk.Logic.Input.AppAction("global.cancel", "Cancel Menu")
            );

            inputRegistry.Register(
              "Viewport",
              new AetherVk.Logic.Input.InputChord(Key: "Escape"),
              new AetherVk.Logic.Input.AppAction("viewport.cancel_add_jet", "Cancel Add Jet")
            );
            inputRegistry.Register(
              "Viewport",
              new AetherVk.Logic.Input.InputChord(Key: "Tab"),
              new AetherVk.Logic.Input.AppAction(
                "viewport.toggle_measuring",
                "Toggle Measuring Mode"
              )
            );
            inputRegistry.Register(
              "Viewport",
              new AetherVk.Logic.Input.InputChord(Key: "R"),
              new AetherVk.Logic.Input.AppAction("viewport.reset_camera", "Reset Camera")
            );
            inputRegistry.Register(
              "Viewport",
              new AetherVk.Logic.Input.InputChord(Key: "D0"),
              new AetherVk.Logic.Input.AppAction("viewport.reset_camera", "Snap Camera Home")
            );
            inputRegistry.Register(
              "Viewport",
              new AetherVk.Logic.Input.InputChord(Key: "NumPad0"),
              new AetherVk.Logic.Input.AppAction("viewport.reset_camera", "Snap Camera Home")
            );
            inputRegistry.Register(
              "Viewport",
              new AetherVk.Logic.Input.InputChord(Key: "F"),
              new AetherVk.Logic.Input.AppAction("viewport.snap_to_selected", "Snap Camera to Selected")
            );
            inputRegistry.Register(
              "Viewport",
              new AetherVk.Logic.Input.InputChord(Key: "S", Alt: true),
              new AetherVk.Logic.Input.AppAction(
                "viewport.open_radial_menu",
                "Open Radial Menu",
                "Opens the radial context menu at the current cursor position"
              )
            );
            inputRegistry.Register(
              "Viewport",
              new AetherVk.Logic.Input.InputChord(Key: "Delete"),
              new AetherVk.Logic.Input.AppAction("viewport.delete", "Delete Selected Object")
            );
            inputRegistry.Register(
              "Viewport",
              new AetherVk.Logic.Input.InputChord(Key: "Back"),
              new AetherVk.Logic.Input.AppAction("viewport.delete", "Delete Selected Object")
            );
            inputRegistry.Register(
              "Viewport",
              new AetherVk.Logic.Input.InputChord(Key: "Up"),
              new AetherVk.Logic.Input.AppAction("viewport.move_cursor_up", "Move Cursor Y-Forward")
            );
            inputRegistry.Register(
              "Viewport",
              new AetherVk.Logic.Input.InputChord(Key: "Down"),
              new AetherVk.Logic.Input.AppAction(
                "viewport.move_cursor_down",
                "Move Cursor Y-Backward"
              )
            );
            inputRegistry.Register(
              "Viewport",
              new AetherVk.Logic.Input.InputChord(Key: "Left"),
              new AetherVk.Logic.Input.AppAction("viewport.move_cursor_left", "Move Cursor X-Left")
            );
            inputRegistry.Register(
              "Viewport",
              new AetherVk.Logic.Input.InputChord(Key: "Right"),
              new AetherVk.Logic.Input.AppAction(
                "viewport.move_cursor_right",
                "Move Cursor X-Right"
              )
            );
            inputRegistry.Register(
              "Viewport",
              new AetherVk.Logic.Input.InputChord(Key: "E"),
              new AetherVk.Logic.Input.AppAction("viewport.move_cursor_z_up", "Move Cursor Z-Up")
            );
            inputRegistry.Register(
              "Viewport",
              new AetherVk.Logic.Input.InputChord(Key: "Q"),
              new AetherVk.Logic.Input.AppAction(
                "viewport.move_cursor_z_down",
                "Move Cursor Z-Down"
              )
            );

            // Blender style camera controls
            inputRegistry.Register(
              "Viewport",
              new AetherVk.Logic.Input.InputChord(Pointer: "MiddleButtonPressed"),
              new AetherVk.Logic.Input.AppAction(
                "viewport.start_orbit",
                "Orbit Camera",
                "Orbit camera around 3D cursor"
              )
            );
            inputRegistry.Register(
              "Viewport",
              new AetherVk.Logic.Input.InputChord(Pointer: "MiddleButtonPressed", Shift: true),
              new AetherVk.Logic.Input.AppAction(
                "viewport.start_pan",
                "Pan Camera",
                "Translate camera along view plane"
              )
            );
            inputRegistry.Register(
              "Viewport",
              new AetherVk.Logic.Input.InputChord(Pointer: "MiddleButtonPressed", Ctrl: true),
              new AetherVk.Logic.Input.AppAction(
                "viewport.start_zoom_drag",
                "Zoom Camera (Drag)",
                "Smooth zoom camera"
              )
            );

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
