using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Shapes;
using System;

namespace AetherVk.UserControls
{
    public sealed partial class SplitContainerControl : UserControl
    {
        // an input or output is a dependency property, which you can set with GeneratedDependencyGenerator
        public SplitContainerControl()
        {
            InitializeComponent();
            Loaded += OnLoaded;
        }

        private void OnLoaded(object sender, RoutedEventArgs e)
        {
            // add Rectagle to both panes
            LeftPanel.Children.Add(new Pages.PanelHostPage());
            RightPanel.Children.Add(CreateRandomRectangle());
        }

        private Rectangle CreateRandomRectangle()
        {
            return new Rectangle
            {
                Fill = new SolidColorBrush(RandomColor()),
                Margin = new Thickness(20)
            };
        }

        private Windows.UI.Color RandomColor()
        {
            return Windows.UI.Color.FromArgb(255, (byte)_Random.Next(), (byte)_Random.Next(), (byte)_Random.Next());
        }

        private readonly Random _Random = new();
    }
}
