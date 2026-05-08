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

    collection.AddSingleton<ConsoleService>();
    collection.AddSingleton<BreadcrumbService>();
    collection.AddSingleton<HorizonJplService>();
    collection.AddSingleton<SceneStateManager>();
    collection.AddSingleton<NativeRuntimeService>();
    collection.AddSingleton<FileWatcherService>();
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
    collection.AddTransient<HorizonJplViewModel>();
    collection.AddTransient<TimelineViewModel>();
    collection.AddTransient<AlmanacExplorerViewModel>();
    collection.AddTransient<Viewport3DViewModel>();
    collection.AddTransient<OutlineViewModel>();
    collection.AddTransient<PropertiesViewModel>();
  }
}
