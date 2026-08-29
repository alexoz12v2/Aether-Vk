using Avalonia.Controls;

using System.Diagnostics;

namespace AetherVk.Views;

public partial class CometTabView : UserControl
{
  public CometTabView()
  {
    InitializeComponent();
    LoadDebugUI();
  }

  [Conditional("DEBUG")]
  private void LoadDebugUI()
  {
    var container = this.FindControl<ContentControl>("DebugContainer");
    if (container != null)
    {
      container.Content = new CometTabDebugView();
    }
  }
}
