using System;
using System.Collections.Generic;

namespace AetherVk.Logic.Services;

/// <inheritdoc cref="ITabStateRegistry"/>
public sealed class TabStateRegistry : ITabStateRegistry
{
  // VM type → non-generic service reference
  private readonly Dictionary<Type, ITabStateService> _byVmType      = new();
  // Session type → non-generic service reference (for GetService<T>)
  private readonly Dictionary<Type, ITabStateService> _bySessionType = new();

  /// <summary>
  /// Called once per stateful tab type at DI setup time.
  /// </summary>
  public void Register<TViewModel, TSession>(ITabStateService<TSession> service)
    where TSession : class, ITabSession, new()
  {
    _byVmType[typeof(TViewModel)]    = service;
    _bySessionType[typeof(TSession)] = service;
  }

  /// <inheritdoc/>
  public bool IsStateful(Type tabViewModelType) =>
    _byVmType.ContainsKey(tabViewModelType);

  /// <inheritdoc/>
  public ITabStateService<TSession> GetService<TSession>()
    where TSession : class, ITabSession, new()
  {
    if (_bySessionType.TryGetValue(typeof(TSession), out var s))
      return (ITabStateService<TSession>)s;
    throw new KeyNotFoundException(
      $"No service is registered for session type '{typeof(TSession).Name}'.");
  }

  /// <inheritdoc/>
  public ITabStateService? TryGetServiceFor(Type tabViewModelType)
  {
    _byVmType.TryGetValue(tabViewModelType, out var s);
    return s;
  }
}
