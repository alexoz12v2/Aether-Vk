using System;
using Avalonia.Controls;
using Avalonia.Controls.Templates;

namespace AetherVk.Utils;

public class ViewLocator : IDataTemplate
{
  private static readonly System.Collections.Generic.Dictionary<Type, Func<Control>> _registry = new()
  {
    [typeof(Logic.ViewModels.MainWindowViewModel)] = () => new MainWindow(),
    [typeof(Logic.ViewModels.SplashViewModel)] = () => new Views.SplashWindow(),
    [typeof(Logic.ViewModels.SettingsViewModel)] = () => new Views.SettingsWindow(),
    [typeof(Logic.ViewModels.DockingManagerViewModel)] = () => new Views.DockingManagerView(),
    [typeof(Logic.ViewModels.Viewport3DViewModel)] = () => new Views.Viewport3DView(),
    [typeof(Logic.ViewModels.TabItemViewModel)] = () => new Views.TabItemView(),
    [typeof(Logic.ViewModels.TabGroupNodeViewModel)] = () => new Views.TabGroupNodeView(),
    [typeof(Logic.ViewModels.HomePageViewModel)] = () => new Views.HomePageView(),
    [typeof(Logic.ViewModels.SplitNodeViewModel)] = () => new Views.SplitNodeView(),
    [typeof(Logic.ViewModels.ConsoleViewModel)] = () => new Views.ConsoleView(),
    [typeof(Logic.ViewModels.VulkanViewportControlViewModel)] = () => new Controls.VulkanViewportControl(),

    // Note: we don't have here registration for "Pure Views", which have StyledProperties and no
    // view model

#if DEBUG
    [typeof(Logic.ViewModels.DebugUiViewModel)] = () => new Views.DebugUiView(),
    [typeof(Logic.ViewModels.UITestPanelViewModel)] = () => new Views.UITestPanelView(),
#endif
  };

  public Control? Build(object? data)
  {
    if (data != null && _registry.TryGetValue(data.GetType(), out var factory))
    {
      return factory();
    }

    return new TextBlock { Text = "Not Found: " + data?.GetType().Name };
  }

  public bool Match(object? data)
  {
    return data is CommunityToolkit.Mvvm.ComponentModel.ObservableObject;
  }
}
