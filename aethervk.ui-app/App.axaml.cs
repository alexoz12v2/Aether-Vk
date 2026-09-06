using System;
using System.Linq;
using System.Threading;
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
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace AetherVk;

public partial class App : Application
{
  public static IHost? Host { get; set; }

  internal static void OnRustPanic(nint messagePtr, nuint length)
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

      WeakReferenceMessenger.Default.Register<App, Logic.Messages.CriticalErrorMessage>(
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

      WeakReferenceMessenger.Default.Register<App, Logic.Messages.CopyToClipboardMessage>(
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

      if (!skipNative)
      {
        var splashViewModel = new SplashViewModel();
        var splashWindow = new Views.SplashWindow { DataContext = splashViewModel };

        splashViewModel.OnInitializationCompleted += () =>
        {
          Avalonia.Threading.Dispatcher.UIThread.Post(() =>
          {
            var inputRegistry =
              ServiceProviderServiceExtensions.GetRequiredService<Logic.Input.InputRegistry>(
                Host!.Services
              );
            // TODO LOAD INPUT BINDIINGS INTO INPUT REGISTRY
            // Configure the singleton before instantiating main window view model

            var mainWindowViewModel =
              ServiceProviderServiceExtensions.GetRequiredService<MainWindowViewModel>(
                Host!.Services
              );
            var mainWindow = new MainWindow { DataContext = mainWindowViewModel };

            // Sync OS theme state for the first-click bug fix
            mainWindowViewModel.IsSystemThemeDark = Current!.ActualThemeVariant == ThemeVariant.Dark;
            Current.ActualThemeVariantChanged += (s, e) =>
            {
                if (mainWindowViewModel.CurrentTheme == AppTheme.System)
                {
                    mainWindowViewModel.IsSystemThemeDark = Current.ActualThemeVariant == ThemeVariant.Dark;
                }
            };

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
        _ = splashViewModel.InitializeAsync(() =>
        {
          return ServiceProviderServiceExtensions.GetRequiredService<INativeRuntimeService>(
            Host!.Services
          );
        });

        var appLifetime =
          ServiceProviderServiceExtensions.GetRequiredService<IHostApplicationLifetime>(
            Host!.Services
          );

        // Link the .NET Generic Host lifetime to Avalonia's lifetime.
        // This ensures that when the generic host receives a SIGINT (e.g., from CTRL+C or dotnet watch),
        // it gracefully tells the Avalonia UI thread to shut down. Otherwise, the app would hang.
        appLifetime.ApplicationStopping.Register(() =>
        {
          Avalonia.Threading.Dispatcher.UIThread.Post(() =>
          {
            desktop.Shutdown();
          });
        });

        desktop.Exit += (sender, args) => {
          // EDIT: I think it's not necessary to call dispose here, cause DI Container should handle
          // that. This block will remain if there are special actions to be performed on closing
          // the application

          // All native services which are disposable should be disposed of here
          // TODO ensure all dependencies which use the runtime service are cleaned up with a shutdown message?
          // runtimeService.Dispose();

          // Note: Avoid calling Environment.Exit(0) here. Doing so will forcefully terminate the process
          // and prevent the generic host in Program.cs from executing its clean shutdown procedure
          // (like host.StopAsync() and host.Dispose() in the finally block).
        };
      }
    }

    base.OnFrameworkInitializationCompleted();
  }
}

#if DEBUG
public class MockNativeRuntimeService : INativeRuntimeService
{
  public void Dispose()
  {
    GC.SuppressFinalize(this);
  }

  public IObservable<ulong> SimulationStateUpdated =>
    System.Reactive.Linq.Observable.Never<ulong>();

  public bool AddViewport(
    uint width,
    uint height,
    string name,
    Func<CNativeWindowHandle>? nativeHandleProvider,
    uint handleType,
    out ulong presentationEngineId,
    out ulong cameraEntityId
  )
  {
    presentationEngineId = 1;
    cameraEntityId = 2;
    return true;
  }

  public void RemoveViewport(ulong presentationEngineId) { }

  public void ResizeViewport(ulong presentationEngineId, uint width, uint height) { }

  public bool TryInitComet(
    int spkId,
    TimeRange proposedRange,
    AetherVk.Logic.Models.SmallBodyDataComponent sbData,
    out ulong cometBodyId
  )
  {
    cometBodyId = 3;
    return true;
  }

  public bool ResetSimulationSync() => true;

  public bool PauseSimulationSync() => true;

  public bool StartSimulation(int simSpeed) => true;

  public bool AddCameraAnimation(ulong cameraId, AnimationTarget animation) => true;

  public bool CameraSetRotoTranslate(
    ulong cameraId,
    System.Numerics.Vector3 position,
    System.Numerics.Quaternion rotation
  ) => true;

  public bool CameraSetPerspective(
    ulong cameraId,
    float fov,
    float aspectRatio,
    float near,
    float far
  ) => true;

  public bool CameraSetOrthographic(
    ulong cameraId,
    float left,
    float right,
    float bottom,
    float top,
    float near,
    float far
  ) => true;

  public bool AddParticleSystem(
    ParticleSystemModel psModel,
    ParticleSystemJet psJet,
    out ulong outPsId
  )
  {
    outPsId = 3;
    return true;
  }

  public ParticleSystemComputedProperties? AddFirstParticleSystem(
    ParticleSystemModel psModel,
    ParticleSystemJet psJet,
    out ulong outPsId
  )
  {
    outPsId = 3;
    return new ParticleSystemComputedProperties(0f, 0f);
  }

  public bool ModifyParticleSystem(
    ulong psId,
    ParticleSystemModel psModel,
    ParticleSystemJet psJet,
    out ParticleSystemComputedProperties outPsComputedProps
  )
  {
    outPsComputedProps = new ParticleSystemComputedProperties(0f, 0f);
    return true;
  }

  public bool RemoveParticleSystem(ulong psId) => true;

  public bool TryInitComet(int spkId, TimeRange proposedRange, out ulong cometBodyId)
  {
    cometBodyId = 0;
    return true;
  }

  public bool ReconfigureComet(int commandFlags, int spkId, out ulong cometBodyId)
  {
    cometBodyId = 0;
    return true;
  }

  public bool SetBodyRotationalModel(ulong cometBodyEntityId, BodyRotationalModelDto dto) => true;

  public Task<ulong> LoadAlmanacFileAsync(string path) => Task.FromResult(4UL);

  public bool UnloadAlmanacFile(string path) => true;

  public Task<ulong> ImportModelAsync(string path) => Task.FromResult(5UL);

  public void UnloadModel(ulong modelId) { }

  public ulong AddScreenSpaceBillboard(string imagePath, ScreenSpaceBillboard billboard) => 6;

  public bool SetScreenSpaceBillboard(ulong entityId, ScreenSpaceBillboard billboard) => true;

  public bool RemoveScreenSpaceBillboard(ulong entityId) => true;

  public bool GetScreenSpaceBillboard(ulong entityId, out ScreenSpaceBillboard outData)
  {
    outData = new ScreenSpaceBillboard(0, 0, 1, 0, 1, 0);
    return true;
  }

  // ── Callbacks & Dispatch ──────────────────────────────────────────────────
  public IDisposable RegisterSimulationListener(
    ulong entityId,
    ulong componentForeignId,
    Action<nint> handler
  ) => System.Reactive.Disposables.Disposable.Empty;

  public IDisposable RegisterExternalStateListener(
    ExternalStateType stateType,
    Action<nint> handler
  ) => System.Reactive.Disposables.Disposable.Empty;

  // ── Cached State ──────────────────────────────────────────────────────────
  public ulong? CameraEntityId => 2UL;
  public ulong? PresentationEngineId => 1UL;
  public ulong? CometEntityId => null;
  public ulong? EarthEntityId => null;

  // Mock never shuts down mid-flight; expose a never-cancelled token.
  public CancellationToken ShutdownToken => CancellationToken.None;

  // ── Timeline ──────────────────────────────────────────────────────────────
  public bool SetEpochRange(short startCenturies, ulong startNs, short endCenturies, ulong endNs) =>
    true;

  public bool CheckAlmanacCoverage(
    int spkId,
    short startCenturies,
    ulong startNs,
    short endCenturies,
    ulong endNs
  ) => true;

  // ── RenderDoc (debug only — always unavailable in the mock) ───────────────
  public bool IsRenderDocAvailable() => false;
  public void TriggerRenderDocCapture() { }
  public bool StartScopedRenderDocCapture(ulong presentationEngineId) => false;

  public void DebugECSPrint(uint entityCount, ulong[] entityIds, uint compCount, ulong[] comps) { }

  public bool GetDebugTelemetryStats(out DebugTelemetryStats stats)
  {
      stats = new DebugTelemetryStats(
          1024 * 1024 * 128,
          1024 * 1024 * 256,
          1024 * 1024 * 64,
          1024 * 1024 * 512,
          1.5,
          2.0,
          0.5
      );
      return true;
  }
}
#endif
