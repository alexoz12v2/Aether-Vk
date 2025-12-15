using AetherVk.Core.ViewModels;
using Microsoft.UI.Xaml.Controls;

namespace AetherVk.Pages
{
    internal sealed partial class EditorPageSettings : Page
    {
        private EditorPageSettingsViewModel ViewModel => (EditorPageSettingsViewModel)DataContext;

        public EditorPageSettings()
        {
            InitializeComponent();
        }
    }
}
