using System;
using System.Linq;
using System.Threading;
using System.Windows.Input;
using AetherVk.Controls;
using Avalonia;
using Avalonia.Automation;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Xunit;

namespace AetherVk.AppTests;

/// <summary>
/// The "?" help control: footer action, accessible name, AutomationId forwarding, paragraphs and
/// the always-pinned flyout.
/// </summary>
public class PropertyHelpButtonTests
{
  private sealed class CountingCommand : ICommand
  {
    public int Executed;
    public object? LastParameter;
    public event System.EventHandler? CanExecuteChanged { add { } remove { } }
    public bool CanExecute(object? parameter) => true;
    public void Execute(object? parameter)
    {
      Executed++;
      LastParameter = parameter;
    }
  }

  private static (Window window, PropertyHelpButton help) Host(PropertyHelpButton help)
  {
    var window = new Window { Content = new StackPanel { Children = { help } }, Width = 400, Height = 300 };
    window.Show();
    Dispatcher.UIThread.RunJobs();
    return (window, help);
  }

  [AvaloniaFact]
  public void ActionButton_HiddenWithoutActionText()
  {
    var (window, help) = Host(new PropertyHelpButton { HelpHeader = "Albedo", HelpContent = "Geometric albedo." });
    help.OpenFlyout();
    Dispatcher.UIThread.RunJobs();

    Assert.True(help.IsFlyoutOpen);
    Assert.False(help.ActionButton.IsVisible);
    window.Close();
  }

  [AvaloniaFact]
  public void ActionButton_ShownWithActionText_ExecutesCommandAndClosesFlyout()
  {
    var command = new CountingCommand();
    var (window, help) = Host(new PropertyHelpButton
    {
      HelpHeader = "Albedo",
      HelpContent = "Geometric albedo.",
      ActionText = "Reset to default",
      ActionCommand = command,
      ActionCommandParameter = 42,
    });
    help.OpenFlyout();
    Dispatcher.UIThread.RunJobs();

    Assert.True(help.ActionButton.IsVisible);
    Assert.Equal("Reset to default", help.ActionButton.Content);

    help.ActionButton.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
    Dispatcher.UIThread.RunJobs();

    Assert.Equal(1, command.Executed);
    Assert.Equal(42, command.LastParameter);
    Assert.False(help.IsFlyoutOpen);
    window.Close();
  }

  [AvaloniaFact]
  public void AccessibleName_IsTheHeader_FallbackHelp_AndToolTipShowsHeader()
  {
    var (window, help) = Host(new PropertyHelpButton { HelpHeader = "Dust density" });
    Assert.Equal("Dust density", AutomationProperties.GetName(help.TriggerButton));
    Assert.Equal("Dust density", ToolTip.GetTip(help.TriggerButton));

    help.HelpHeader = "Dust albedo"; // reactive
    Assert.Equal("Dust albedo", AutomationProperties.GetName(help.TriggerButton));

    help.HelpHeader = string.Empty;
    Assert.Equal(PropertyHelpButton.DefaultAccessibleName, AutomationProperties.GetName(help.TriggerButton));
    window.Close();
  }

  [AvaloniaFact]
  public void AutomationId_ForwardedToInnerButtonAndAction_OwnerLookupStillWorks()
  {
    var help = new PropertyHelpButton { HelpHeader = "Albedo", ActionText = "Open" };
    AutomationProperties.SetAutomationId(help, "DustAlbedoHelp");
    var (window, _) = Host(help);

    var inner = help.GetVisualDescendants().OfType<Button>()
      .Single(b => AutomationProperties.GetAutomationId(b) == "DustAlbedoHelp");
    Assert.Same(help.TriggerButton, inner);
    Assert.Equal("DustAlbedoHelp" + PropertyHelpButton.ActionAutomationIdSuffix,
      AutomationProperties.GetAutomationId(help.ActionButton));

    // the lookup the dust-parameter view tests use (OfType<PropertyHelpButton> by id) is unchanged
    var owner = window.GetVisualDescendants().OfType<PropertyHelpButton>()
      .Single(h => AutomationProperties.GetAutomationId(h) == "DustAlbedoHelp");
    Assert.Same(help, owner);
    window.Close();
  }

  [AvaloniaFact]
  public void Paragraphs_SplitOnBlankLines_SingleNewlinesKept()
  {
    var help = new PropertyHelpButton { HelpContent = "First line\nsame paragraph\n\nSecond\r\n\r\nThird" };
    Assert.Equal(new[] { "First line\nsame paragraph", "Second", "Third" }, help.Paragraphs);

    // a literal backslash-n typed in a XAML attribute counts too
    Assert.Equal(new[] { "A", "B" }, PropertyHelpButton.SplitParagraphs(@"A\n\nB"));
    Assert.Empty(PropertyHelpButton.SplitParagraphs("  "));
  }

  [AvaloniaFact]
  public void FlyoutMaxWidthAndPlacement_DefaultsAndOverrides()
  {
    var help = new PropertyHelpButton();
    Assert.Equal(240, help.FlyoutMaxWidth);
    Assert.Equal(PlacementMode.RightEdgeAlignedTop, help.FlyoutPlacement);

    help.FlyoutPlacement = PlacementMode.BottomEdgeAlignedLeft;
    Assert.Equal(PlacementMode.BottomEdgeAlignedLeft, ((Flyout)help.TriggerButton.Flyout!).Placement);
  }

  [AvaloniaFact]
  public void KeyboardAndPointerOpensAreBothPinned()
  {
    var (window, help) = Host(new PropertyHelpButton { HelpHeader = "Albedo", HelpContent = "x" });
    help.OpenFlyout();
    Dispatcher.UIThread.RunJobs();
    Assert.True(help.IsFlyoutPinned);
    help.CloseFlyout();
    Dispatcher.UIThread.RunJobs();
    Assert.False(help.IsFlyoutOpen);
    Assert.False(help.IsFlyoutPinned);

    // a real click on the ?
    var centre = TriggerCentre(window, help);
    window.MouseDown(centre, MouseButton.Left);
    window.MouseUp(centre, MouseButton.Left);
    Dispatcher.UIThread.RunJobs();
    Assert.True(help.IsFlyoutOpen);
    Assert.True(help.IsFlyoutPinned);
    window.Close();
  }

  /// <summary>
  /// On Win32 opening the popup can raise a PointerExited on the ? (and clear IsPointerOver) while
  /// the mouse has not moved; the click-opened flyout must survive it and any grace delay.
  /// </summary>
  [AvaloniaFact]
  public void PointerOpen_SpuriousPointerExited_FlyoutStaysOpen()
  {
    var (window, help) = Host(new PropertyHelpButton { HelpHeader = "Albedo", HelpContent = "x" });
    var centre = TriggerCentre(window, help);
    window.MouseDown(centre, MouseButton.Left);
    window.MouseUp(centre, MouseButton.Left);
    Dispatcher.UIThread.RunJobs();
    Assert.True(help.IsFlyoutOpen);

    var pointer = new Pointer(Pointer.GetNextFreeId(), PointerType.Mouse, true);
    var exited = new PointerEventArgs(InputElement.PointerExitedEvent, help.TriggerButton, pointer, window, centre,
      0, new PointerPointProperties(), KeyModifiers.None);
    help.TriggerButton.RaiseEvent(exited);
    Dispatcher.UIThread.RunJobs();

    // well past the 350 ms pointer-away grace the flyout used to have
    var until = DateTime.UtcNow + TimeSpan.FromMilliseconds(800);
    while (DateTime.UtcNow < until)
    {
      Thread.Sleep(50);
      Dispatcher.UIThread.RunJobs();
    }
    Assert.True(help.IsFlyoutOpen);
    Assert.True(help.IsFlyoutPinned);

    // an explicit close still works
    help.CloseFlyout();
    Dispatcher.UIThread.RunJobs();
    Assert.False(help.IsFlyoutOpen);
    window.Close();
  }

  private static Point TriggerCentre(Window window, PropertyHelpButton help)
  {
    var b = help.TriggerButton;
    return b.TranslatePoint(new Point(b.Bounds.Width / 2, b.Bounds.Height / 2), window)!.Value;
  }
}
