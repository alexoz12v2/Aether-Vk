using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Core.Types
{
    public interface IAbstractFactory<T>
    {
        T Create();
    }

    public sealed class AbstractFactory<T>(Func<T> factory) : IAbstractFactory<T>
    {
        public T Create()
        {
            return factory();
        }
    }

    public interface IAbstractParamFactory<TView, TViewModel>
    {
        TView Create(TViewModel viewModel);
    }

    public sealed class AbstractParamFactory<TView, TViewModel>(Func<TViewModel, TView> factory) : IAbstractParamFactory<TView, TViewModel>
    {
        public TView Create(TViewModel viewModel)
        {
            return factory(viewModel);
        }
    }

    public interface IPageFactory<TViewModel>
    {
        object Create(TViewModel self);
    }
}
