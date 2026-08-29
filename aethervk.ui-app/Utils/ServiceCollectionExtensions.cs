using System;
using System.Globalization;
using System.Resources;
using AetherVk.Input;
using AetherVk.Logic.Input;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using AetherVk.Services;
using Microsoft.Extensions.DependencyInjection;

namespace AetherVk.Utils;

public static class ServiceCollectionExtensions
{
  public static void AddCommonServices(this IServiceCollection collection, bool skipNative = false)
  {
    collection.AddSingleton<IUiThreadDispatcher, AvaloniaUiThreadDispatcher>();
    collection.AddSingleton<IWindowService, AvaloniaWindowService>();
    collection.AddSingleton<IFileDialogService, AvaloniaFileDialogService>();
    collection.AddSingleton<ILocalStorageService, LocalStorageService>();
    collection.AddSingleton<ConsoleService>();
    collection.AddSingleton<BreadcrumbService>();
    collection.AddSingleton<HorizonJplService>();
    collection.AddSingleton<INativeBufferPoolService, NativeBufferPoolService>();
    collection.AddSingleton<ISchedulerProvider, AvaloniaSchedulerProvider>();
    // Session registry must be ready before TabFactory is resolved (TabFactory depends on ITabStateRegistry)
    collection.AddTabSessions();
    collection.AddSingleton<ITabFactory, TabFactory>();
#if DEBUG
    if (skipNative)
    {
      collection.AddSingleton<INativeRuntimeService, MockNativeRuntimeService>();
    }
    else
#endif
    {
      collection.AddSingleton<INativeRuntimeService, NativeRuntimeService>(provider =>
      {
        return new NativeRuntimeService(
          provider.GetRequiredService<IUiThreadDispatcher>(),
          provider.GetRequiredService<ConsoleService>(),
          provider.GetRequiredService<BreadcrumbService>(),
          App.OnRustPanic
        );
      });
      // Companion runtime services — resolve in dependency order so the DI container
      // can wire TimelineService → CometPositionTrackerService → CameraService.
      collection.AddSingleton<TimelineService>();
      collection.AddSingleton<CometPositionTrackerService>();
      collection.AddSingleton<ImportedModelsTrackerService>();
      collection.AddSingleton<CameraService>();
      collection.AddSingleton<CometConfigService>();
    }
    collection.AddSingleton<INativeInputHandlerFactory, NativeInputHandlerFactory>();
    collection.AddSingleton<IWindowInputRouter, GlobalInputRouter>();
    collection.AddSingleton<IPlatformWindowService, PlatformWindowService>();
    collection.AddSingleton<IViewportRegistry, ViewportRegistry>();
    var inputRegistry = new InputRegistry();
    inputRegistry.RegisterViewportDefaults();
    collection.AddSingleton<InputRegistry>(_ => inputRegistry);
    collection.AddSingleton<ITranslationService, TranslationService>(provider =>
    {
      // if this is used elsewhere then add it as a singleton
      var resourceManager = new ResourceManager("AetherVk.Logic.Resources.AppStrings", typeof(ITranslationService).Assembly);
      // TODO probably taken from global/local file config?
      var startCulture = new CultureInfo("en");
      return new TranslationService(resourceManager, startCulture);
    });
  }

  public static void AddViewModels(this IServiceCollection collection)
  {
    collection.AddTransient<HomePageViewModel>();
    collection.AddSingleton<DockingManagerViewModel>();
    collection.AddSingleton<MainWindowViewModel>();
    collection.AddTransient<SplashViewModel>();

    // Tabs
    collection.AddTransient<ImportsTabViewModel>();
    collection.AddTransient<TimelineTabViewModel>();
    collection.AddTransient<CometTabViewModel>();
    collection.AddTransient<ModelTabViewModel>();
    collection.AddTransient<SettingsTabViewModel>();
    collection.AddTransient<UITestPanelViewModel>();
    collection.AddTransient<ConsoleViewModel>();
    collection.AddTransient<DebugUiViewModel>();
    collection.AddTransient<Viewport3DViewModel>();
    collection.AddTransient<Func<Viewport3DViewModel, VulkanViewportControlViewModel>>(sp =>
    {
      var router = sp.GetRequiredService<IWindowInputRouter>();
      var factory = sp.GetRequiredService<INativeInputHandlerFactory>();
      var runtime = sp.GetRequiredService<INativeRuntimeService>();
      return vm => new VulkanViewportControlViewModel(router, factory, runtime, vm);
    });
    collection.AddTransient<Func<Viewport3DViewModel, ViewportOverlayViewModel>>(sp =>
    {
      var cameraService     = sp.GetRequiredService<CameraService>();
      var runtimeService    = sp.GetRequiredService<INativeRuntimeService>();
      var breadcrumbService = sp.GetRequiredService<BreadcrumbService>();
      var dispatcher        = sp.GetRequiredService<IUiThreadDispatcher>();
      var fileDialog        = sp.GetRequiredService<IFileDialogService>();
      return vm => new ViewportOverlayViewModel(cameraService, runtimeService, breadcrumbService, dispatcher, fileDialog, vm);
    });
  }

  /// <summary>
  /// Registers <see cref="ITabStateRegistry"/> and one <see cref="ITabStateService{TSession}"/>
  /// singleton per stateful tab type. Must be called before <c>ITabFactory</c> is registered.
  /// </summary>
  public static void AddTabSessions(this IServiceCollection collection)
  {
    // Registry is built lazily on first resolve; the factory sets up all services at that point.
    collection.AddSingleton<ITabStateRegistry>(sp =>
    {
      var schedulers = sp.GetRequiredService<ISchedulerProvider>();
      var registry = new TabStateRegistry();

      TabStateService<T> MakeService<T>() where T : class, ITabSession, new()
        => new TabStateService<T>(schedulers);

      var cometSvc = MakeService<CometSession>();
      var modelSvc = MakeService<ModelSession>();
      var settingsSvc = MakeService<SettingsSession>();
      var importsSvc = MakeService<ImportsSession>();
      var timelineSvc = MakeService<TimelineSession>();
      var viewportSvc = MakeService<ViewportSession>();

      registry.Register<CometTabViewModel, CometSession>(cometSvc);
      registry.Register<ModelTabViewModel, ModelSession>(modelSvc);
      registry.Register<SettingsTabViewModel, SettingsSession>(settingsSvc);
      registry.Register<ImportsTabViewModel, ImportsSession>(importsSvc);
      registry.Register<TimelineTabViewModel, TimelineSession>(timelineSvc);
      registry.Register<Viewport3DViewModel, ViewportSession>(viewportSvc);

      // Stash instances so individual ITabStateService<T> registrations below can return them.
      _pendingCometSvc = cometSvc;
      _pendingModelSvc = modelSvc;
      _pendingSettingsSvc = settingsSvc;
      _pendingImportsSvc = importsSvc;
      _pendingTimelineSvc = timelineSvc;
      _pendingViewportSvc = viewportSvc;

      return registry;
    });

    // Each ITabStateService<T> delegates to the already-built registry singleton so the
    // same object is always returned regardless of resolution order.
    collection.AddSingleton<ITabStateService<CometSession>>(
      sp => { sp.GetRequiredService<ITabStateRegistry>(); return _pendingCometSvc!; });
    collection.AddSingleton<ITabStateService<ModelSession>>(
      sp => { sp.GetRequiredService<ITabStateRegistry>(); return _pendingModelSvc!; });
    collection.AddSingleton<ITabStateService<SettingsSession>>(
      sp => { sp.GetRequiredService<ITabStateRegistry>(); return _pendingSettingsSvc!; });
    collection.AddSingleton<ITabStateService<ImportsSession>>(
      sp => { sp.GetRequiredService<ITabStateRegistry>(); return _pendingImportsSvc!; });
    collection.AddSingleton<ITabStateService<TimelineSession>>(
      sp => { sp.GetRequiredService<ITabStateRegistry>(); return _pendingTimelineSvc!; });
    collection.AddSingleton<ITabStateService<ViewportSession>>(
      sp => { sp.GetRequiredService<ITabStateRegistry>(); return _pendingViewportSvc!; });
  }

  // Holding fields — set once inside the ITabStateRegistry factory lambda, effectively immutable after DI build.
  private static ITabStateService<CometSession>? _pendingCometSvc;
  private static ITabStateService<ModelSession>? _pendingModelSvc;
  private static ITabStateService<SettingsSession>? _pendingSettingsSvc;
  private static ITabStateService<ImportsSession>? _pendingImportsSvc;
  private static ITabStateService<TimelineSession>? _pendingTimelineSvc;
  private static ITabStateService<ViewportSession>? _pendingViewportSvc;
}
