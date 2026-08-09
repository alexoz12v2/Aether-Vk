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
    collection.AddSingleton<ITabFactory, TabFactory>();
    collection.AddSingleton<ILocalStorageService, LocalStorageService>();
    collection.AddSingleton<ConsoleService>();
    collection.AddSingleton<BreadcrumbService>();
    collection.AddSingleton<HorizonJplService>();
    collection.AddSingleton<INativeBufferPoolService, NativeBufferPoolService>();
    collection.AddSingleton<ISchedulerProvider, AvaloniaSchedulerProvider>();
#if DEBUG
    if (skipNative)
    {
      collection.AddSingleton<INativeRuntimeService, MockNativeRuntimeService>();
    }
    else
#endif
    {
      // TODO do it
      // collection.AddSingleton<INativeRuntimeService, NativeRuntimeService>();
    }
    collection.AddSingleton<INativeInputHandlerFactory, NativeInputHandlerFactory>();
    collection.AddSingleton<IWindowInputRouter, GlobalInputRouter>();
    collection.AddSingleton<InputRegistry>();
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
    collection.AddTransient<VulkanViewportControlViewModel>();
  }
}
