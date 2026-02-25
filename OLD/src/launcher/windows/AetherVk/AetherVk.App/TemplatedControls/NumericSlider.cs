using CommunityToolkit.WinUI;
using Microsoft.UI.Xaml.Controls;

namespace AetherVk.TemplatedControls
{
    public sealed partial class NumericSlider : Control
    {
        public NumericSlider()
        {
            DefaultStyleKey = typeof(NumericSlider);
        }

        [GeneratedDependencyProperty(DefaultValue = 0d)]
        public partial double Minimum { get; set; }

        [GeneratedDependencyProperty(DefaultValue = 100.0)]
        public partial double Maximum { get; set; }

        [GeneratedDependencyProperty(DefaultValue = 50.0)]
        public partial double Value { get; set; }
    }

}