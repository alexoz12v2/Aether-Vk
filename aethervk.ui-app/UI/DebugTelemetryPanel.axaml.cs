using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Markup.Xaml;
using AetherVk.Logic.ViewModels.Debug;

namespace AetherVk.UI;

public partial class DebugTelemetryPanel : UserControl
{
    public DebugTelemetryPanel()
    {
        InitializeComponent();
    }

    private void InitializeComponent()
    {
        AvaloniaXamlLoader.Load(this);
    }

    /// <summary>
    /// Toggles <see cref="DebugTelemetryPanelViewModel.IsExpanded"/> immediately on
    /// <c>PointerPressed</c> rather than waiting for <c>PointerReleased</c>.
    ///
    /// The <see cref="OverlayWindow"/> is a separate, non-activating top-level window.
    /// On Linux the WM reassigns focus to the main window as soon as the user clicks the
    /// overlay, which causes Avalonia to fire <c>PointerCaptureLost</c> before the matching
    /// <c>PointerReleased</c> event arrives.  The Expander's inner <c>ToggleButton</c>
    /// responds only to <c>PointerReleased</c>, so its toggle logic never executes.
    ///
    /// By handling the event here on <c>PointerPressed</c> and marking it as handled, we
    /// prevent the <c>ToggleButton</c>'s own handler from seeing a half-completed press, and
    /// the expansion state is committed before the capture can be lost.
    /// </summary>
    private void OnHeaderPointerPressed(object? sender, PointerPressedEventArgs e)
    {
        if (DataContext is DebugTelemetryPanelViewModel vm)
        {
            vm.IsExpanded = !vm.IsExpanded;
            e.Handled = true;
        }
    }
}
