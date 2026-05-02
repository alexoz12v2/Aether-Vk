using System;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using Avalonia.Threading;

namespace AetherVk.Services
{
    public class AvaloniaUiThreadDispatcher : IUiThreadDispatcher
    {
        public void Dispatch(Action action)
        {
            Dispatcher.UIThread.Post(action);
        }

        public Task DispatchAsync(Func<Task> action)
        {
            return Dispatcher.UIThread.InvokeAsync(action);
        }
    }
}
