using System;
using System.Collections.Generic;
using System.Linq;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace AetherVk.Services;

public class TabFactory : ITabFactory
{
  private readonly IServiceProvider    _serviceProvider;
  private readonly ITabStateRegistry   _registry;

  private static readonly Dictionary<Type, (string Header, Func<IServiceProvider, object> Factory)> _tabMap;

  static TabFactory()
  {
    _tabMap = new()
    {
      [typeof(ConsoleViewModel)]     = ("Console",     sp => sp.GetRequiredService<ConsoleViewModel>()),
      [typeof(Viewport3DViewModel)]  = ("Viewport 3D", sp => sp.GetRequiredService<Viewport3DViewModel>()),
      [typeof(SettingsTabViewModel)] = ("Settings",    sp => sp.GetRequiredService<SettingsTabViewModel>()),
      [typeof(CometTabViewModel)]    = ("Comet",       sp => sp.GetRequiredService<CometTabViewModel>()),
      [typeof(ModelTabViewModel)]    = ("Model",       sp => sp.GetRequiredService<ModelTabViewModel>()),
      [typeof(ImportsTabViewModel)]  = ("Imports",     sp => sp.GetRequiredService<ImportsTabViewModel>()),
      [typeof(TimelineTabViewModel)] = ("Timeline",    sp => sp.GetRequiredService<TimelineTabViewModel>()),
    };

#if DEBUG
    _tabMap[typeof(UITestPanelViewModel)] = ("UI Test Panel", sp => sp.GetRequiredService<UITestPanelViewModel>());
    _tabMap[typeof(DebugUiViewModel)]     = ("Debug UI",      sp => sp.GetRequiredService<DebugUiViewModel>());
#endif
  }

  public TabFactory(IServiceProvider serviceProvider, ITabStateRegistry registry)
  {
    _serviceProvider = serviceProvider;
    _registry        = registry;
  }

  public object? CreateTab(Type tabType)
  {
    if (_tabMap.TryGetValue(tabType, out var entry))
      return entry.Factory(_serviceProvider);
    return null;
  }

  public T? CreateTab<T>() where T : class => CreateTab(typeof(T)) as T;

  public (object? ViewModel, Action? Dispose) CreateScopedTab(Type tabType)
  {
    if (!_tabMap.TryGetValue(tabType, out var entry))
      return (null, null);

    var scope = _serviceProvider.CreateScope();
    var vm    = entry.Factory(scope.ServiceProvider);
    return (vm, scope.Dispose);
  }

  public IReadOnlyList<TabDescriptor> AvailableTabs =>
    _tabMap
      .Select(kv => new TabDescriptor(
        kv.Value.Header,
        kv.Key,
        _registry.IsStateful(kv.Key)))
      .ToList();
}
