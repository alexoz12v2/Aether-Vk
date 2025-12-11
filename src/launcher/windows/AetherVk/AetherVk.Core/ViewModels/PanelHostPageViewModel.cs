using AetherVk.Core.Types;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System.Collections.ObjectModel;
using System.Diagnostics;

namespace AetherVk.Core.ViewModels
{
    public sealed partial class PanelHostPageViewModel : ObservableObject
    {
        [ObservableProperty]
        public partial ObservableCollection<EditorDescriptor> Editors { get; set; }

        public PanelHostPageViewModel()
        {
            Editors = [
                new EditorDescriptor(label: "Splash Screen", pageType: EditorType.SplashScreen, SelectEditorCommand) {
                Glyph = "\uE80F" // Home glyph
            }.EnsureValid()
            ];
        }

        [RelayCommand]
        private void SelectEditor(EditorType? editorType)
        {
            Debug.WriteLine("Hello Beautiful World");
        }
    }
}
