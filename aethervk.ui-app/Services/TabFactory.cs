using System;
using System.Collections.Generic;
using System.Linq;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace AetherVk.Services;

public class TabFactory(IServiceProvider serviceProvider) : ITabFactory
{
  private readonly IServiceProvider _serviceProvider = serviceProvider;

  private static readonly Dictionary<Type, (string Header, Func<IServiceProvider, object> Factory)> _registry;

  static TabFactory()
  {
    _registry = new()
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
    _registry[typeof(UITestPanelViewModel)] = ("UI Test Panel", sp => sp.GetRequiredService<UITestPanelViewModel>());
    _registry[typeof(DebugUiViewModel)]     = ("Debug UI",      sp => sp.GetRequiredService<DebugUiViewModel>());
#endif
  }

  public object? CreateTab(Type tabType)
  {
    if (_registry.TryGetValue(tabType, out var value))
      return value.Factory(_serviceProvider);
    return null;
  }

  public T? CreateTab<T>() where T : class => CreateTab(typeof(T)) as T;

  public IReadOnlyList<TabDescriptor> AvailableTabs { get; } =
    _registry.Select(kv => new TabDescriptor(kv.Value.Header, kv.Key)).ToList();
}
