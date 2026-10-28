using AetherVk.Logic.ViewModels;
using Avalonia.Controls;

namespace AetherVk.Views;

public partial class TimelineTabView : UserControl
{
  public TimelineTabView()
  {
    InitializeComponent();
    // playback must not overwrite an epoch being typed
    CurrentEpochBox.GotFocus += (_, _) => SetEditing(true);
    CurrentEpochBox.LostFocus += (_, _) => SetEditing(false);
  }

  private void SetEditing(bool editing)
  {
    if (DataContext is TimelineTabViewModel vm)
      vm.IsEditingCurrentEpoch = editing;
  }
}
