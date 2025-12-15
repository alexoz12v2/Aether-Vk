using AetherVk.Core.Private;
using AetherVk.Core.Types;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;
using System.Collections.ObjectModel;
using System.Diagnostics;

// Idle
//  └─ pointer near border ─▶ Hover
// Hover
//  ├─ pointer leaves ──────▶ Idle
//  └─ LMB down ───────────▶ Dragging
// Dragging
//  ├─ pointer move ───────▶ Dragging (update preview)
//  ├─ RMB / Esc ─────────▶ Cancelled
//  └─ LMB up ────────────▶ Commit
// Commit / Cancel
//  └─ cleanup ───────────▶ Idle

// | State    | Owner         | Notes                  |
// | -------- | ------------- | ---------------------- |
// | Idle     | View          | No VM involvement      |
// | Hover    | View          | Visual affordance only |
// | Dragging | **ViewModel** | Intent exists          |
// | Commit   | ViewModel     | Mutates layout         |
// | Cancel   | ViewModel     | No mutation            |


namespace AetherVk.Core.ViewModels
{
    public sealed partial class PanelHostPageViewModel : ObservableRecipient
    {
        public ReadOnlyCollection<EditorDescriptor> Editors { get; set; }

        [ObservableProperty]
        public partial EditorType SelectedEditor { get; set; } = EditorType.SplashScreen;

        public PanelHostPageViewModel(IMessenger splitLayoutMessenger) : base(splitLayoutMessenger)
        {
            Debug.Assert(ReferenceEquals(Messenger, splitLayoutMessenger));

            Editors = [
                new EditorDescriptor(label: "Splash Screen", pageType: EditorType.SplashScreen, SelectEditorCommand) {
                    Glyph = "\uE80F" // Home glyph
                }.EnsureValid(),
                new EditorDescriptor(label: "Console", pageType: EditorType.Console, SelectEditorCommand) {
                    Glyph = "\uE756" // commandPrompt glyph
                }.EnsureValid()
            ];
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

        #region SplitMessages
        public void BeginSplitSession(Point start, RectD bounds)
        {
            _ = Messenger.Send(new SplitSessionBegin(this, start, bounds));
        }

        public void UpdateSplitSession(Point current)
        {
            _ = Messenger.Send(new SplitSessionUpdate(this, current));
        }

        public void EndSplitSession(Point end, bool cancelled)
        {
            _ = Messenger.Send(new SplitSessionEnd(this, end, cancelled));
        }
        #endregion
    }
}
