using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace AetherVk.Views;

public partial class TabItemView : UserControl
{
    public TabItemView()
    {
        InitializeComponent();
    }

    private void InitializeComponent()
    {
        AvaloniaXamlLoader.Load(this);
    }
}
