using System;
using AetherVk.Logic.Input;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using AetherVk.Services;
using Microsoft.Extensions.DependencyInjection;

namespace AetherVk.Utils;

public static class ServiceCollectionExtensions
{
  public static void AddCommonServices(this IServiceCollection collection)
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
    // TODO do it
    // collection.AddSingleton<INativeRuntimeService, NativeRuntimeService>();
    collection.AddSingleton<INativeInputHandlerFactory, NativeInputHandlerFactory>();
    collection.AddSingleton<InputRegistry>();
  }

  public static void AddViewModels(this IServiceCollection collection)
  {
    collection.AddTransient<HomePageViewModel>();
    collection.AddSingleton<DockingManagerViewModel>();
    collection.AddSingleton<MainWindowViewModel>();
    collection.AddTransient<SplashViewModel>();

    // Tabs
    collection.AddTransient<UITestPanelViewModel>();
    collection.AddTransient<ConsoleViewModel>();
    collection.AddTransient<DebugUiViewModel>();
    collection.AddTransient<Viewport3DViewModel>();
    collection.AddTransient<VulkanViewportControlViewModel>();

    // Tab Factories
    collection.AddSingleton<Func<UITestPanelViewModel>>(sp =>
      () => sp.GetRequiredService<UITestPanelViewModel>()
    );
    collection.AddSingleton<Func<ConsoleViewModel>>(sp =>
      () => sp.GetRequiredService<ConsoleViewModel>()
    );
    collection.AddSingleton<Func<DebugUiViewModel>>(sp =>
      () => sp.GetRequiredService<DebugUiViewModel>()
    );
    collection.AddSingleton<Func<Viewport3DViewModel>>(sp =>
      () => sp.GetRequiredService<Viewport3DViewModel>()
    );
    collection.AddSingleton<Func<VulkanViewportControlViewModel>>(sp =>
      () => sp.GetRequiredService<VulkanViewportControlViewModel>()
    );
  }
}
