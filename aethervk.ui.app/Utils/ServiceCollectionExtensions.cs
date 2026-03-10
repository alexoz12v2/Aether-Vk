using System;
using AetherVk.Logic.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace AetherVk.Utils;

public static class ServiceCollectionExtensions
{
    public static void AddCommonServices(this IServiceCollection collection)
    {
        // TODO Logging and stuff
    }

    public static void AddViewModels(this IServiceCollection collection)
    {
        collection.AddTransient<HomePageViewModel>();
    }
}