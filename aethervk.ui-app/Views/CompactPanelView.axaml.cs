using AetherVk.Logic.ViewModels;
using Avalonia.Controls;

namespace AetherVk.Views;

/// <summary>The compact layout's panel (see CompactPanelView.axaml).</summary>
public partial class CompactPanelView : UserControl
{
  public CompactPanelView()
  {
    InitializeComponent();
    // playback must not overwrite an epoch being typed
    EpochBox.GotFocus += (_, _) => SetEditing(true);
    EpochBox.LostFocus += (_, _) => SetEditing(false);
  }

  private void SetEditing(bool editing)
  {
    if (DataContext is CompactPanelViewModel { Timeline: { } timeline })
      timeline.IsEditingCurrentEpoch = editing;
  }
}
