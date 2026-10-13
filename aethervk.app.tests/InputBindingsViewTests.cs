#if DEBUG
using System.Linq;
using AetherVk.Logic.ViewModels;
using AetherVk.Views;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Input;
using Avalonia.LogicalTree;
using Avalonia.Threading;
using Xunit;

namespace AetherVk.AppTests;

/// <summary>The Settings "Input bindings" editor captures real key presses (docs/savefile.md).</summary>
public class InputBindingsViewTests
{
  [AvaloniaFact]
  public void Rebind_CapturesTheNextKeyChord_AndEscCancels()
  {
    var vm = new DesignSettingsTabViewModel();
    var view = new SettingsTabView { DataContext = vm };
    var window = new Window { Content = view, Width = 600, Height = 1600 };
    window.Show();
    view.GetLogicalDescendants().OfType<Expander>().Single(e => e.Name == "InputBindingsSection").IsExpanded = true;
    Dispatcher.UIThread.RunJobs();

    // as in the app: the clicked Rebind button holds the focus, so key events reach the view
    var rebind = view.GetLogicalDescendants().OfType<Button>()
      .Single(b => b.Classes.Contains("binding-rebind") && Equals(b.Content, "V"));
    rebind.Focus();
    rebind.Command!.Execute(rebind.CommandParameter);
    Assert.True(vm.InputBindings.IsCapturing);
    // a lone modifier does not end the capture
    window.KeyPress(Key.LeftShift, RawInputModifiers.Shift, PhysicalKey.ShiftLeft, null);
    Assert.True(vm.InputBindings.IsCapturing);
    window.KeyPress(Key.C, RawInputModifiers.Shift, PhysicalKey.C, "C");
    Dispatcher.UIThread.RunJobs();
    Assert.False(vm.InputBindings.IsCapturing);
    var chords = vm.InputBindings.Contexts[0].Actions.Single(a => a.ActionId == "viewport.switch_camera_mode")
      .Chords.Select(c => c.Text).OrderBy(t => t).ToList();
    Assert.Equal(["Shift + C", "Tab"], chords);

    // Esc cancels a capture
    var add = view.GetLogicalDescendants().OfType<Button>().First(b => b.Command == vm.InputBindings.Contexts[0].Actions[0].AddChordCommand);
    add.Focus();
    add.Command!.Execute(null);
    window.KeyPress(Key.Escape, RawInputModifiers.None, PhysicalKey.Escape, null);
    Dispatcher.UIThread.RunJobs();
    Assert.False(vm.InputBindings.IsCapturing);
    window.Close();
  }
}
#endif
