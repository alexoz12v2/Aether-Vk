using System;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace AetherVk.Utils;

public static class ServiceCollectionExtensions
{
  public static void AddCommonServices(this IServiceCollection collection)
  {
    collection.AddSingleton<ConsoleService>();
    collection.AddSingleton<BreadcrumbService>();
    collection.AddSingleton<HorizonJplService>();
    collection.AddSingleton<NativeRuntimeService>();
  }

  public static void AddViewModels(this IServiceCollection collection)
  {
    collection.AddTransient<HomePageViewModel>();
  }
}
