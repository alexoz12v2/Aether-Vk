using System;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using AetherVk.Services;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Interactivity;
using Avalonia.LogicalTree;
using Avalonia.Threading;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace AetherVk.AppTests;

/// <summary>
/// The savefile's window side, headless: the main window's title follows the view model (the
/// dirty marker), the File menu is built in the real window, the unsaved-changes dialog returns
/// the clicked choice, and closing with unsaved changes asks first (headless, the prompt has no
/// desktop owner and answers Cancel, so the window stays open).
/// </summary>
public class FileMenuViewTests
{
  [AvaloniaFact]
  public void MainWindow_ShowsTheFileMenu_TitleAndDirtyMarker_AndAsksBeforeClosing()
  {
    var services = (App.Host ?? throw new InvalidOperationException("DI host not initialized")).Services;
    var vm = services.GetRequiredService<MainWindowViewModel>();
    if (services.GetService<Logic.SaveFile.SaveFileService>() is null)
      return; // mocked native runtime: no savefile services
    var models = services.GetRequiredService<ITabStateService<ModelSession>>();
    var model = models.GetSession(models.ActiveSessionIds[0]);

    var window = new MainWindow { DataContext = vm };
    window.Show();
    Dispatcher.UIThread.RunJobs();
    vm.RefreshTitle();
    Assert.StartsWith("AetherVk — ", window.Title);
    Assert.Equal(vm.WindowTitle, window.Title);

    var file = window.GetLogicalDescendants().OfType<MenuItem>().Single(m => Equals(m.Header, "File"));
    var headers = file.Items.OfType<MenuItem>().Select(m => m.Header as string).ToList();
    Assert.Equal(["New", "Open...", "Open Recent", "Save", "Save As...", "Save As Folder..."], headers);

    // an edit marks the title, and closing asks (headless: Cancel) instead of closing
    var before = model.DiametreUm;
    model.DiametreUm = before + 123;
    vm.RefreshTitle();
    Assert.EndsWith(" *", window.Title);
    window.Close();
    Dispatcher.UIThread.RunJobs();
    Assert.True(window.IsVisible, "closed despite unsaved changes");

    model.DiametreUm = before;
    vm.RefreshTitle();
    Assert.False(window.Title!.EndsWith(" *", StringComparison.Ordinal));
    window.Close();
    Dispatcher.UIThread.RunJobs();
    Assert.False(window.IsVisible);
  }

  [AvaloniaTheory]
  [InlineData("Save", UnsavedChoice.Save)]
  [InlineData("Discard", UnsavedChoice.Discard)]
  [InlineData("Cancel", UnsavedChoice.Cancel)]
  public async Task UnsavedChangesDialog_ReturnsTheClickedChoice(string button, UnsavedChoice expected)
  {
    var owner = new Window();
    owner.Show();
    var dialog = AvaloniaUnsavedChangesPrompt.BuildDialog("67P study");
    var result = dialog.ShowDialog<UnsavedChoice?>(owner);
    Dispatcher.UIThread.RunJobs();
    var text = dialog.GetLogicalDescendants().OfType<TextBlock>().First().Text;
    Assert.Contains("67P study", text);
    var b = dialog.GetLogicalDescendants().OfType<Button>().Single(x => x.Name == button);
    b.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
    Assert.Equal(expected, await result);
    owner.Close();
  }
}
