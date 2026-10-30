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

  /// <summary>The per-second perf lines (AETHERVK_PERF_LOG=1), kept alive for the app's lifetime.</summary>
  private static PerfLogService? _perfLog;

  internal static void OnRustPanic(nint messagePtr, nuint length)
  {
    string errorMsg = "Unknown Rust Panic";
    if (messagePtr != IntPtr.Zero)
    {
      errorMsg =
        System.Runtime.InteropServices.Marshal.PtrToStringAnsi(messagePtr, (int)length) ?? errorMsg;
    }

    // The core aborts the process after this returns, so the panic must be on disk first
    var logSink = Host?.Services.GetService<LogFileSink>();
    logSink?.Write($"[FATAL] Rust core panicked: {errorMsg}");
    logSink?.Flush(TimeSpan.FromSeconds(2));

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

      var debugWindow = new Views.Debug.ShaderDebugWindow();
      debugWindow.Show();
      
      AetherVk.Logic.Services.NativeRuntimeService.ShaderDebugLogAction = debugWindow.AppendLog;

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
          // Startup is over: from here a silent log is not a hang
          Host?.Services.GetService<StartupWatchdog>()?.Disarm();

          Avalonia.Threading.Dispatcher.UIThread.Post(() =>
          {
            var inputRegistry =
              ServiceProviderServiceExtensions.GetRequiredService<Logic.Input.InputRegistry>(
                Host!.Services
              );
            // preferences.json: the user's bindings replace the defaults, the theme is restored
            // (before the main window view model reads either)
            var preferences = ServiceProviderServiceExtensions.GetRequiredService<PreferencesService>(Host!.Services);
            preferences.Load();
            _ = inputRegistry;
            // the dust volume detail and temporal reuse tolerance (Settings › Particle rendering)
            // are preferences: the scene starts at 16 / off, the stored values go to it before the
            // first frame
            var runtimeAtStartup = ServiceProviderServiceExtensions.GetRequiredService<INativeRuntimeService>(Host!.Services);
            runtimeAtStartup.SetDustVolumeDetail(preferences.DustVolumeDetail);
            runtimeAtStartup.SetDustPropagateDetail(preferences.DustPropagateDetail);
            runtimeAtStartup.SetDustHazeDetail(preferences.DustHazeDetail);
            runtimeAtStartup.SetDustReuseTolerance(preferences.DustReuseTolerance);
            runtimeAtStartup.SetDustPlayPreview(preferences.DustPlayPreview);
            // leap seconds: the validity horizon starts from today; a cached newer kernel loads
            // now, the NAIF check runs in the background (never with the mock runtime)
            runtimeAtStartup.SetNowUtc(DateTimeOffset.UtcNow);
            bool mockRuntime = false;
#if DEBUG
            mockRuntime = runtimeAtStartup is MockNativeRuntimeService;
#endif
            if (!mockRuntime)
            {
              var leapSeconds = ServiceProviderServiceExtensions.GetRequiredService<LeapSecondsUpdateService>(Host!.Services);
              leapSeconds.LoadCachedKernel();
              _ = leapSeconds.StartBackgroundUpdate(
                ServiceProviderServiceExtensions.GetRequiredService<IHostApplicationLifetime>(Host!.Services).ApplicationStopping);
              // the playback e2e's per-second perf lines (AETHERVK_PERF_LOG=1)
              if (PerfLogService.Enabled)
              {
                _perfLog = new PerfLogService(runtimeAtStartup);
                _perfLog.Start();
              }
            }

            var mainWindowViewModel =
              ServiceProviderServiceExtensions.GetRequiredService<MainWindowViewModel>(
                Host!.Services
              );
            UI.WindowFrameTheme.Install(Current!,
              ServiceProviderServiceExtensions.GetRequiredService<IPlatformWindowService>(Host!.Services));
            var mainWindow = new MainWindow { DataContext = mainWindowViewModel };

            // Sync OS theme state for the first-click bug fix
            mainWindowViewModel.IsSystemThemeDark =
              Current!.ActualThemeVariant == ThemeVariant.Dark;
            Current.ActualThemeVariantChanged += (s, e) =>
            {
              if (mainWindowViewModel.CurrentTheme == AppTheme.System)
              {
                mainWindowViewModel.IsSystemThemeDark =
                  Current.ActualThemeVariant == ThemeVariant.Dark;
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

            mainWindowViewModel.CurrentTheme = preferences.Theme;
            Current!.RequestedThemeVariant = preferences.Theme switch
            {
              AppTheme.Light => ThemeVariant.Light,
              AppTheme.Dark => ThemeVariant.Dark,
              _ => ThemeVariant.Default,
            };
            mainWindowViewModel.PropertyChanged += (_, a) =>
            {
              if (a.PropertyName == nameof(MainWindowViewModel.CurrentTheme))
                preferences.SetTheme(mainWindowViewModel.CurrentTheme);
            };

            desktop.MainWindow = mainWindow;
            mainWindow.Closing += (_, e) => { if (!e.Cancel) DetachViewport(); };

            mainWindow.Show();
            splashWindow.Close();
            // the end-to-end suite's crash-report check (needs AETHERVK_E2E=1 as well)
            Logic.Services.CrashReport.StartTestCrashIfRequested(Environment.GetEnvironmentVariable, TimeSpan.FromSeconds(3));
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

        // detach the viewport while its window still exists: presenting to a destroyed window
        // deadlocked the render thread at shutdown ("presentation engines are still attached")
        desktop.ShutdownRequested += (_, _) => DetachViewport();

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

  /// <summary>
  /// Removes the viewport's presentation engine while its window still exists: presenting to a
  /// destroyed window deadlocked the render thread at shutdown. Idempotent.
  /// </summary>
  private void DetachViewport()
  {
    var runtime = Host?.Services.GetService<INativeRuntimeService>();
    if (runtime?.PresentationEngineId is ulong pe)
      runtime.RemoveViewport(pe);
  }
}

#if DEBUG
public class MockNativeRuntimeService : INativeRuntimeService
{
  public void Dispose()
  {
    GC.SuppressFinalize(this);
  }


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

  public bool GetSimulationClock(out short centuries, out ulong nanoseconds)
  {
    centuries = 0;
    nanoseconds = 0;
    return true;
  }

  // UTC ↔ TDB without the core: a fixed 69.184 s (ΔAT 37 s + 32.184 s), no leap-second table
  private const double MockTdbMinusUtcSeconds = 69.184;

  public UtcStatus UtcToTdb(UtcFields utc, out short centuries, out ulong nanoseconds)
  {
    (centuries, nanoseconds) = AetherVk.Logic.Utils.TimeUtils.ToTdbParts(utc.ToDateTimeOffset().AddSeconds(MockTdbMinusUtcSeconds));
    return utc.Year < 1972 ? UtcStatus.BeforeLeapSeconds : UtcStatus.Ok;
  }

  public UtcStatus TdbToUtc(short centuries, ulong nanoseconds, out UtcFields utc)
  {
    utc = UtcFields.From(AetherVk.Logic.Utils.TimeUtils.FromTdbParts(centuries, nanoseconds).AddSeconds(-MockTdbMinusUtcSeconds));
    return utc.Year < 1972 ? UtcStatus.BeforeLeapSeconds : UtcStatus.Ok;
  }

  public bool SetNowUtc(DateTimeOffset now) => true;

  public (LeapSecondsLoadStatus Status, int EntryCount) LoadLeapSecondsFile(string path) => (LeapSecondsLoadStatus.KeptCurrent, 0);

  public LeapSecondsInfo LeapSecondsInfo() => new(false, 28, 37, new DateTime(2017, 1, 1, 0, 0, 0, DateTimeKind.Utc), string.Empty);

  public bool CleanupParticleSystem(ulong psId) => true;

  public bool SetJetPreviewVisibility(ulong psId, bool visibility) => true;
  public bool GetJetLitFraction(ulong jetEntityId, out double litFraction, out TimeRange committedRange)
  {
    litFraction = 1;
    committedRange = new TimeRange(0, 0, 0, 0);
    return false;
  }
  public bool SetJetAngleVisibility(ulong psId, bool visibility) => true;

  public bool SetCometIndicatorVisible(bool visible) => true;
  public bool SetCameraRenderMask(ulong cameraEntityId, uint mask) => true;
  public bool SetCameraShading(ulong cameraEntityId, uint flags) => true;
  public bool SetCameraExposure(ulong cameraEntityId, float factor) => true;
  public CometScreenPosition? GetCometScreenPosition(ulong cameraEntityId, uint widthPx, uint heightPx) => null;
  public Logic.Models.AssetThumbnail? CopyAssetTexels(ulong assetId) => null;
  public Logic.Models.IntensityImage? CopyFitsRaster(ulong assetId) => null;
  public string? GetFitsMetadataJson(ulong assetId) => null;
  public Logic.Models.FitsStretchResult? SetFitsStretch(ulong assetId, uint mode, uint transfer, double p0, double p1) => null;
  public bool DumpSceneToFile(string file) => true;
  public bool RestoreSceneFromFile(string file) => true;
  public string? GetCompatibilityJson() => "{}";
  public Logic.Models.ExportedAssetFile? ExportAsset(ulong assetId, string destDir) => null;
  public ulong? GetAssetContentHash(ulong assetId) => null;
  public uint SceneDumpVersion => 0;
  public double DustComaRadiusKm() => 0.0;
  public EarthObserverStatus? EarthObserverStatus() => null;
  public CometEphemeris? CometEphemeris() => null;
  public bool SetEarthObserver(ulong cameraId, int mode, ulong earthId, ulong cometId, bool topocentric, double latDeg, double lonDeg, double elevM, AetherVk.Logic.Utils.Quaterniond look, AetherVk.Logic.Services.ObserverUp up) => true;
  public bool SetNucleusViewFromEarth(ulong cameraId, ulong earthId, ulong cometId, double distanceAu) => true;

  public bool SetReferenceErrorVisible(bool visible) => true;
  public bool SetDustSoftening(float softening) => true;
  public bool SetDustVolumeDetail(int detail) => true;
  public bool SetDustPropagateDetail(int detail) => true;
  public bool SetDustHazeDetail(int detail) => true;
  public bool SetDustReuseTolerance(int px) => true;
  public bool SetDustPlayPreview(int level) => true;
  public bool DustRenderFull() => true;
  public bool? DustInPreview() => null;
  public (DustRenderPhase Phase, float HoldProgress)? DustRenderState() => null;
  public bool ResetDustViewFeedback() => true;
  public bool SetDustViewFlags(uint flags) => true;
  public bool SetDustExposureEv(float ev) => true;
  public bool SetDustFlowSpeed(float speed) => true;

  public void RemoveViewport(ulong presentationEngineId) { }

  public void ResizeViewport(ulong presentationEngineId, uint width, uint height) { }

  public bool TryInitComet(
    int spkId,
    TimeRange proposedRange,
    Logic.Models.SmallBodyDataComponent sbData,
    Logic.Models.ReferenceOrbitMode referenceMode,
    out ulong cometBodyId
  )
  {
    cometBodyId = 3;
    return true;
  }

  public bool ResetSimulationSync() => true;

  public bool SeekEpochSync(short centuries, ulong nanoseconds) => true;

  public bool DumpScene(string baseDir) => true;

  public bool RestoreScene(string baseDir) => true;

  public bool PauseSimulationSync() => true;

  public bool StartSimulation(int simSpeed) => true;
  
  public bool SnapshotSceneSync() => true;
  
  public bool RestoreSnapshotSync() => true;

  public bool DebugCameraState(
    ulong cameraEntityId,
    out double posX,
    out double posY,
    out double posZ,
    out double rotX,
    out double rotY,
    out double rotZ,
    out double rotW
  )
  {
    posX = posY = posZ = 0.0;
    rotX = rotY = rotZ = rotW = 0.0;
    return true;
  }

  public bool AddCameraAnimation(ulong cameraId, AnimationTarget animation) => true;

  public System.Collections.Generic.IReadOnlyList<DustTierStats> DustStats() =>
    System.Array.Empty<DustTierStats>();

  public DustExposure? DustExposure(ulong presentationEngineId) => null;

  public DustGpuTimings? DustGpuTimings(ulong presentationEngineId) => null;

  public DustGpuTimings? DustGpuTimingsDrawn(ulong presentationEngineId) => null;

  public bool CameraSetRotoTranslate(
    ulong cameraId,
    double posX,
    double posY,
    double posZ,
    AetherVk.Logic.Utils.Quaterniond rotation,
    ulong pivotEntityId = 0
  ) => true;

  public bool SetCameraParent(ulong cameraId, ulong parentEntityId, bool enabled) => true;
  public bool SetCameraParentToComet(ulong cameraId, bool enabled) => true;

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

  public bool UpdateCometNucleusRadius(float radiusKm) => true;

  public bool SetSphereGizmoVisibility(bool isVisible) => true;
  public bool SetSphereGizmoParts(uint parts) => true;

  public Task<ulong> LoadAlmanacFileAsync(string path) => Task.FromResult(4UL);

  public bool UnloadAlmanacFile(string path) => true;

  public Task<ulong> ImportModelAsync(string path) => Task.FromResult(5UL);

  public void UnloadModel(ulong modelId) { }

  public bool ImportAsset(ulong requestId, string path, string cacheDir) => false;

  public bool RemoveAsset(ulong requestId, ulong assetId) => false;

  public AetherVk.Logic.Models.ImportedAsset[]? GetAssets() => [];

  public AetherVk.Logic.Models.AssetThumbnail? GetAssetThumbnail(ulong assetId) => null;

  public AetherVk.Logic.Models.CometAppearanceStatus SetCometAppearance(
    AetherVk.Logic.Models.CometAppearanceDto appearance
  ) => AetherVk.Logic.Models.CometAppearanceStatus.NotAvailable;

  public bool GetCometAppearance(out AetherVk.Logic.Models.CometAppearanceDto appearance)
  {
    appearance = default;
    return false;
  }

  public SkyImageInfo? LoadSkyImage(string path, string? sidecarPath) => null;

  public bool SetSkyImageParams(float opacity, float stretch, bool visible) => false;

  public bool RemoveSkyImage() => false;

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
