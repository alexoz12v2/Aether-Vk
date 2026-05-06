using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace AetherVk.Views;

public partial class UITestPanelView : UserControl
{
  public UITestPanelView()
  {
    InitializeComponent();
  }

  private void InitializeComponent()
  {
    AvaloniaXamlLoader.Load(this);
  }
}
