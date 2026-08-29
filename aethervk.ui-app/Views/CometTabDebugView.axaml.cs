using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace AetherVk.Views;

public partial class CometTabDebugView : UserControl
{
    public CometTabDebugView()
    {
        InitializeComponent();
    }

    private void InitializeComponent()
    {
        AvaloniaXamlLoader.Load(this);
    }
}
