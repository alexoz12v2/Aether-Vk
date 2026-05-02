using System;
using System.Threading.Tasks;

namespace AetherVk.Logic.Services
{
    public interface IUiThreadDispatcher
    {
        void Dispatch(Action action);
        Task DispatchAsync(Func<Task> action);
    }
}
