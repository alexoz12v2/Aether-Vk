
using AetherVk.Core.Types;
using System.Collections.ObjectModel;

namespace AetherVk.Core.Interfaces
{
    public interface IApp
    {
        void Exit();
    }

    // Needed if ViewModel or other classes need to interact with the DispatcherQueue without depending on the Windows SDK
    public interface IUIDispatcher
    {
        void Enqueue(Action action);
    }

    public interface IEditorDictionaryService
    {
        IReadOnlyCollection<string> GetEditorNames();
        IReadOnlyCollection<EditorInfo> GetEditors();
        IReadOnlyDictionary<string, Type> GetEditorTypes();
    }
}