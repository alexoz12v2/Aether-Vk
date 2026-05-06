using System;
using System.Reflection;
using Avalonia.Controls;
using Avalonia.Controls.Templates;

namespace AetherVk.Utils;

public class ViewLocator : IDataTemplate
{
  public Control? Build(object? data)
  {
    if (data is null)
      return null;

    var name = data.GetType().FullName!;

    // Correctly replace the ViewModel namespace with the View namespace
    var viewName = name.Replace("AetherVk.Logic.ViewModels", "AetherVk.Views");
    viewName = viewName.Replace("ViewModel", "View", StringComparison.Ordinal);

    // Get the currently executing assembly (AetherVk.UI.App) which contains the Views
    var assembly = Assembly.GetExecutingAssembly();
    var type = assembly.GetType(viewName);

    if (type != null)
    {
      return (Control)Activator.CreateInstance(type)!;
    }

    return new TextBlock { Text = "Not Found: " + viewName };
  }

  public bool Match(object? data)
  {
    // This locator is intended for ViewModels.
    return data?.GetType().FullName?.EndsWith("ViewModel") ?? false;
  }
}
