
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
}