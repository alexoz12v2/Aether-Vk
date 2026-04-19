using System;

namespace AetherVk.Logic.Services;

public static class ServiceLocator
{
  public static IServiceProvider? Provider { get; set; }
  public static Action<Action>? DispatchToUI { get; set; }
}
