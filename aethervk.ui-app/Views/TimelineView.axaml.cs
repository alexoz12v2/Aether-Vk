using AetherVk.Logic.ViewModels;
using Avalonia.Controls;
using Avalonia.Input;

namespace AetherVk.Views;

public partial class TimelineView : UserControl
{
  public TimelineView()
  {
    InitializeComponent();
  }

  private void OnSliderPointerPressed(object? sender, PointerPressedEventArgs e)
  {
    if (DataContext is TimelineViewModel vm)
    {
      vm.BeginDrag();
    }
  }

  private void OnSliderPointerReleased(object? sender, PointerReleasedEventArgs e)
  {
    if (DataContext is TimelineViewModel vm)
    {
      vm.EndDrag();
    }
  }
}
