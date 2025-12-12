using AetherVk.Core.Types;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;
using System.Collections.ObjectModel;
using System.Diagnostics;

namespace AetherVk.Core.ViewModels
{
    public sealed partial class PanelHostPageViewModel : ObservableRecipient
    {
        public ReadOnlyCollection<EditorDescriptor> Editors { get; set; }

        [ObservableProperty]
        public partial EditorType SelectedEditor { get; set; } = EditorType.SplashScreen;

        private readonly IMessenger _SplitLayoutMessenger;

        public PanelHostPageViewModel(IMessenger splitLayoutMessenger) : base(splitLayoutMessenger)
        {
            _SplitLayoutMessenger = splitLayoutMessenger;

            Editors = [
                new EditorDescriptor(label: "Splash Screen", pageType: EditorType.SplashScreen, SelectEditorCommand) {
                    Glyph = "\uE80F" // Home glyph
                }.EnsureValid(),
                new EditorDescriptor(label: "Console", pageType: EditorType.Console, SelectEditorCommand) {
                    Glyph = "\uE756" // commandPrompt glyph
                }.EnsureValid()
            ];
            // TODO: register to some message
        }

        [RelayCommand]
        private void SelectEditor(EditorType? editorType)
        {
            if (editorType.HasValue && SelectedEditor != editorType.Value)
            {
                Debug.WriteLine($"Changing Selected Editor From {SelectedEditor} to {editorType.Value}");
                SelectedEditor = editorType.Value;
            }
        }
    }
}
