using Avalonia.Controls;
using Avalonia.Markup.Xaml;

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
}
