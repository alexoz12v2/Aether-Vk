using System;
using System.Collections.Generic;
using AetherVk.Logic.ViewModels;

namespace AetherVk.Logic.Services;

public interface ITabFactory
{
  object? CreateTab(Type tabType);
  T? CreateTab<T>() where T : class;

  /// <summary>
  /// Creates a tab ViewModel inside a new DI scope.
  /// The returned <see cref="Action"/> disposes the scope (and thus the ViewModel)
  /// when invoked. The caller MUST invoke it when the tab is closed.
  /// </summary>
  (object? ViewModel, Action? Dispose) CreateScopedTab(Type tabType);

  IReadOnlyList<TabDescriptor> AvailableTabs { get; }
}

