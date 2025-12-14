
using AetherVk.Core.Types;
using Microsoft.Extensions.DependencyInjection;
using System;

namespace AetherVk.Launch
{
    internal static class Extensions
    {
        // DependencyObjects don't have an interface as we don't needed
        public static void AddFactory<T>(this IServiceCollection services)
            where T : class
        {
            _ = services.AddTransient<T>();
            _ = services.AddSingleton<Func<T>>(x => () => x.GetService<T>()!);
            _ = services.AddSingleton<IAbstractFactory<T>, AbstractFactory<T>>();
        }
    }
}
