using AetherVk.Core.Types;
using AetherVk.Core.ViewModels;
using Microsoft.UI;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Documents;
using Microsoft.UI.Xaml.Media;
using System.Threading.Tasks;

namespace AetherVk.Pages
{
    internal sealed partial class EditorPageConsole : Page
    {
        private EditorPageConsoleViewModel ViewModel => (EditorPageConsoleViewModel)DataContext;

        public EditorPageConsole()
        {
            InitializeComponent();

            Loaded += EditorPageConsole_OnLoaded;
        }

        private void EditorPageConsole_OnLoaded(object sender, RoutedEventArgs e)
        {
            _ = DispatcherQueue.TryEnqueue(() => _ = AddText());
        }

        private async Task AddText()
        {
            await Task.Delay(2000);
            Paragraph paragraph = new();
            paragraph.Inlines.Add(new Run { Text = "The beautiful Text", Foreground = new SolidColorBrush(Colors.Green) });
            ConsoleContent.Blocks.Add(paragraph);
            _ = DispatcherQueue.TryEnqueue(() => _ = AddText());
        }
    }
}
