using AetherVk.Core.ViewModels;
using Microsoft.UI.Xaml.Controls;

namespace AetherVk.Pages
{
    public sealed partial class EditorPageSplashScreen : Page
    {
        private EditorPageSplashScreenViewModel ViewModel => (EditorPageSplashScreenViewModel)DataContext;

        public EditorPageSplashScreen()
        {
            InitializeComponent();
        }

    }
}
