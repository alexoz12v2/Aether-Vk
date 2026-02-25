using AetherVk.Core.Interfaces;
using AetherVk.Core.Private;
using AetherVk.Core.Types;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;
using System.Collections.ObjectModel;
using System.ComponentModel.Design;
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
        public ReadOnlyCollection<EditorDescriptor> Editors { get; private set; }

        [ObservableProperty]
        public partial string SelectedEditor { get; set; }

        private readonly IEditorDictionaryService _DictionaryService;

        public PanelHostPageViewModel(IMessenger splitLayoutMessenger, IEditorDictionaryService dictionaryService) : base(splitLayoutMessenger)
        {
            Debug.Assert(ReferenceEquals(Messenger, splitLayoutMessenger));
            _DictionaryService = dictionaryService;
            Editors = [.. dictionaryService.GetEditors().Select(info => new EditorDescriptor(info, SelectEditorCommand))];
            SelectedEditor = Editors[0].PageType;
        }

        public IReadOnlyDictionary<string, Type> GetEditorPages()
        {
            return _DictionaryService.GetEditorTypes();
        }

        [RelayCommand]
        private void SelectEditor(string editorType)
        {
            if (SelectedEditor != editorType)
            {
                Debug.WriteLine($"Changing Selected Editor From {SelectedEditor} to {editorType}");
                SelectedEditor = editorType;
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
