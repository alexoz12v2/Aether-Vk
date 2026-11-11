using System;
using System.Globalization;
using System.Linq;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.LogicalTree;
using Avalonia.Threading;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace AetherVk.AppTests;

/// <summary>
/// The Layout menu in the real window, headless: it follows Edit in the menu bar, holds the two
/// radio entries Classic and Compact of one group, Classic is checked in a classic session, and
/// the view model's checked state reaches the menu items.
/// </summary>
public class LayoutMenuViewTests
{
  [AvaloniaFact]
  public void MainWindow_HasTheLayoutMenu_WithClassicAndCompactRadios()
  {
    var services = (App.Host ?? throw new InvalidOperationException("DI host not initialized")).Services;
    var vm = services.GetRequiredService<MainWindowViewModel>();
    var translation = services.GetRequiredService<ITranslationService>();
    var culture = translation.CurrentCulture;
    // the shared host may carry a layout or a language left by earlier tests
    services.GetRequiredService<UiLayoutService>().Set(UiLayout.Classic);
    translation.CurrentCulture = new CultureInfo("en");
    Dispatcher.UIThread.RunJobs();
    try
    {
      var window = new MainWindow { DataContext = vm };
      window.Show();
      Dispatcher.UIThread.RunJobs();

      var menu = window.GetLogicalDescendants().OfType<Menu>().Single();
      var top = menu.Items.OfType<MenuItem>().Select(m => m.Header as string).ToList();
      if (services.GetService<Logic.SaveFile.SaveFileService>() is not null)
        Assert.Equal(["File", "Edit", "Layout"], top);
      else
        Assert.Equal(["Edit", "Layout"], top); // mocked native runtime: no File menu

      var layout = menu.Items.OfType<MenuItem>().Single(m => Equals(m.Header, "Layout"));
      var items = layout.Items.OfType<MenuItem>().ToList();
      Assert.Equal(["Classic", "Compact"], items.Select(m => m.Header as string));
      Assert.All(items, m => Assert.Equal(MenuItemToggleType.Radio, m.ToggleType));
      Assert.All(items, m => Assert.Equal(MainWindowViewModel.UiLayoutGroupName, m.GroupName));
      Assert.True(items[0].IsChecked, "Classic is checked in a classic session");
      Assert.False(items[1].IsChecked);

      // the view model's checked state reaches the menu items
      var vmItems = vm.MainMenu.Single(m => m.Header == "Layout").Items;
      vmItems[0].IsChecked = false;
      vmItems[1].IsChecked = true;
      Dispatcher.UIThread.RunJobs();
      Assert.False(items[0].IsChecked);
      Assert.True(items[1].IsChecked);
      vmItems[1].IsChecked = false;
      vmItems[0].IsChecked = true;
      Dispatcher.UIThread.RunJobs();
      Assert.True(items[0].IsChecked);
      Assert.False(items[1].IsChecked);

      window.Close();
      Dispatcher.UIThread.RunJobs();
    }
    finally
    {
      translation.CurrentCulture = culture;
      Dispatcher.UIThread.RunJobs();
    }
  }
}
