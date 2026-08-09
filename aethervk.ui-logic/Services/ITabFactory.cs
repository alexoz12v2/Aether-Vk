using System;
using System.Collections.Generic;
using AetherVk.Logic.ViewModels;

namespace AetherVk.Logic.Services;

public interface ITabFactory
{
  object? CreateTab(Type tabType);
  T? CreateTab<T>() where T : class;
  IReadOnlyList<TabDescriptor> AvailableTabs { get; }
}

