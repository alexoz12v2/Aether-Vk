using System;

namespace AetherVk.Logic.Services;

/// <summary>
/// Central registry that maps tab VM types to their <see cref="ITabStateService{TSession}"/> instances.
/// Populated at DI registration time via <c>ServiceCollectionExtensions.AddTabSessions()</c>.
/// </summary>
public interface ITabStateRegistry
{
  /// <summary>Returns <c>true</c> when <paramref name="tabViewModelType"/> participates in the session system.</summary>
  bool IsStateful(Type tabViewModelType);

  /// <summary>Retrieves the service for a given session type.</summary>
  ITabStateService<TSession> GetService<TSession>()
    where TSession : class, ITabSession, new();

  /// <summary>
  /// Returns the non-generic service handle for the given VM type, or <c>null</c> if the type is stateless.
  /// </summary>
  ITabStateService? TryGetServiceFor(Type tabViewModelType);
}
