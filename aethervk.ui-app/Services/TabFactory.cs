using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace AetherVk.Services;

public class TabFactory(IServiceProvider serviceProvider) : ITabFactory
{
  private readonly IServiceProvider _serviceProvider = serviceProvider;

  private static readonly Dictionary<Type, (string Header, Func<IServiceProvider, object> Factory)> _registry = new()
  {
    [typeof(UITestPanelViewModel)] = ("UI Test Panel", sp => sp.GetRequiredService<UITestPanelViewModel>()),
    [typeof(ConsoleViewModel)] = ("Console", sp => sp.GetRequiredService<ConsoleViewModel>()),
    [typeof(DebugUiViewModel)] = ("Debug UI", sp => sp.GetRequiredService<DebugUiViewModel>()),
    [typeof(Viewport3DViewModel)] = ("Viewport 3D", sp => sp.GetRequiredService<Viewport3DViewModel>())
    // Add other view models here when they are properly implemented
  };

  public object? CreateTab(Type tabType)
  {
    if (_registry.TryGetValue(tabType, out var value))
    {
      return value.Factory(_serviceProvider);
    }
    return null;
  }

  public T? CreateTab<T>() where T : class
  {
    return CreateTab(typeof(T)) as T;
  }

  public IReadOnlyList<TabDescriptor> AvailableTabs { get; } = [.. _registry.Select(kv => new TabDescriptor(kv.Value.Header, kv.Key))];
}

