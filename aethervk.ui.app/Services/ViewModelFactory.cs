using System;
using AetherVk.Logic.Services;
using Microsoft.Extensions.DependencyInjection;

namespace AetherVk.Services
{
    public class ViewModelFactory : IViewModelFactory
    {
        private readonly IServiceProvider _serviceProvider;

        public ViewModelFactory(IServiceProvider serviceProvider)
        {
            _serviceProvider = serviceProvider;
        }

        public object CreateViewModel(string tabType)
        {
            var type = Type.GetType(tabType);
            if (type == null)
            {
                throw new ArgumentException($"Cannot find type {tabType}");
            }
            return ActivatorUtilities.CreateInstance(_serviceProvider, type);
        }
    }
}
