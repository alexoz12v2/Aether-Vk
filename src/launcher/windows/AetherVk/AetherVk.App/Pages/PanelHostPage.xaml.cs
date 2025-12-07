using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace AetherVk.Pages
{
    internal sealed partial class PanelHostPage : Page
    {
        public PanelHostPage()
        {
            InitializeComponent();
        }

        private void HeaderFlyout_Opened(object sender, object e)
        {
            HeaderIcon.Glyph = (string)Application.Current.Resources["GlyphChevronUp"];
        }

        private void HeaderFlyout_Closed(object sender, object e)
        {
            HeaderIcon.Glyph = (string)Application.Current.Resources["GlyphChevronDown"];
        }
    }
}
