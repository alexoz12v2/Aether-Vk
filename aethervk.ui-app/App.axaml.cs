using System;
using System.Linq;
using System.Threading.Tasks;
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
      bool skipNative = false;
#if DEBUG
      skipNative = desktop.Args?.Contains("--skip-native") == true;

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

      if (!skipNative && !System.IO.File.Exists(libPath) && !System.IO.File.Exists(libName))
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

#if DEBUG
public class MockNativeRuntimeService : INativeRuntimeService
{
  public void Dispose() { GC.SuppressFinalize(this); }
  public bool Startup() => true;
  public void ShutdownSync() { }
  public bool AddViewport(uint width, uint height, string name, out ulong presentationEngineId, out ulong cameraEntityId)
  {
    presentationEngineId = 1;
    cameraEntityId = 2;
    return true;
  }
  public void RemoveViewport(ulong presentationEngineId) { }
  public void ResizeViewport(ulong presentationEngineId, uint width, uint height) { }
  public Task<bool> DownloadImageAsync(ulong taskId, IntPtr bufferPtr, nuint bufferSize) => Task.FromResult(true);
  public bool ResetSimulationSync() => true;
  public bool PauseSimulationSync() => true;
  public bool StartSimulation(int simSpeed) => true;
  public bool ModifyComponent(ulong entityId, uint command, IntPtr inDto, IntPtr outComputedDto) => true;
  public bool AddCameraAnimation(ulong cameraId, ref AnimationTargetDTO animation) => true;
  public bool TransformStaticCamera(ulong cameraId, int mode, IntPtr buffer) => true;
  public bool AddParticleSystem(ref ParticleSystemDTO particleSystem, out ulong outPsId)
  {
    outPsId = 3;
    return true;
  }
  public bool ModifyParticleSystem(ulong psId, ref ParticleSystemDTO particleSystem, out ParticleSystemComputedDTO outPsComputedProps)
  {
    outPsComputedProps = new ParticleSystemComputedDTO();
    return true;
  }
  public bool ReconfigureComet() => true;
  public Task<ulong> LoadAlmanacFileAsync(string path) => Task.FromResult(4UL);
  public bool UnloadAlmanacFile(string path) => true;
  public void SetAssetPath(string path) { }
  public Task<ulong> ImportModelAsync(string path) => Task.FromResult(5UL);
  public void UnloadModel(ulong modelId) { }
  public ulong AddScreenSpaceBillboard(string imagePath, float ndcX, float ndcY, float scale, float rotationDeg, float opacity, int zIndex, ulong viewportId) => 6;
  public bool SetScreenSpaceBillboard(ulong entityId, float ndcX, float ndcY, float scale, float rotationDeg, float opacity, int zIndex) => true;
  public bool RemoveScreenSpaceBillboard(ulong entityId) => true;
  public bool GetScreenSpaceBillboard(ulong entityId, out FfiScreenSpaceBillboardDTO outData)
  {
    outData = new FfiScreenSpaceBillboardDTO();
    return true;
  }
  public void RegisterPanicCallback(PanicCallbackDelegate cb) { }
  public void SetLoggerCallback(LoggerCallbackDelegate cb) { }
  public void SetBreadcrumbCallback(BreadcrumbCallbackDelegate cb) { }
  public void SetSimulationCallback(SimulationCallbackDelegate cb) { }
  public void SetExternalStateSimulationCallback(ExternalStateSimulationCallbackDelegate cb) { }
  public void SetRenderCallback(RenderCallbackDelegate cb) { }
  public void SetMainThreadDispatchCallback(MainThreadDispatchCallbackDelegate cb) { }
}
#endif
