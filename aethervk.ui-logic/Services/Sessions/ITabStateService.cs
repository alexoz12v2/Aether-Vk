using System;
using System.Collections.Generic;

namespace AetherVk.Logic.Services;

/// <summary>
/// Non-generic marker interface so <see cref="ITabStateRegistry"/> can hold heterogeneous services.
/// </summary>
public interface ITabStateService { }

/// <summary>
/// Per-session-type singleton that manages numbered session instances.
/// <list type="bullet">
///   <item>Observables are delivered on the UI scheduler (Avalonia-safe for binding).</item>
///   <item>All writes go through <see cref="UpdateSession"/> — the single guarded mutation path.</item>
///   <item>Sessions outlive their tabs. Closing a tab never deletes its session.</item>
///   <item>The last remaining session cannot be deleted; a fresh one is created first.</item>
/// </list>
/// </summary>
public interface ITabStateService<TSession> : ITabStateService
  where TSession : class, ITabSession, new()
{
  /// <summary>
  /// <c>true</c> when the session type is marked with <see cref="ExclusiveSessionAttribute"/>
  /// and therefore at most one session may ever exist.
  /// </summary>
  bool IsExclusive { get; }

  /// <summary>Snapshot of all currently active session IDs.</summary>
  IReadOnlyList<SessionId> ActiveSessionIds { get; }

  /// <summary>
  /// Hot observable that fires the current list immediately on subscribe and on every change.
  /// Delivered on the main-thread scheduler.
  /// </summary>
  IObservable<IReadOnlyList<SessionId>> ObserveSessionList();

  /// <summary>
  /// Hot, replay-1 observable for the session's data.
  /// Delivers the latest value immediately on subscribe and on every <see cref="UpdateSession"/> call.
  /// Delivered on the main-thread scheduler.
  /// </summary>
  IObservable<TSession> ObserveSession(SessionId id);

  /// <summary>Returns the current (non-reactive) snapshot of a session's data.</summary>
  TSession GetSession(SessionId id);

  /// <summary>
  /// Allocates a new session with the next auto-incremented number.
  /// </summary>
  /// <exception cref="InvalidOperationException">
  /// Thrown when <see cref="IsExclusive"/> is <c>true</c> and a session already exists.
  /// </exception>
  SessionId CreateSession();

  /// <summary>
  /// Removes the specified session.
  /// If this would delete the last session, a fresh replacement is created first so that
  /// at least one session is always alive.
  /// </summary>
  void DeleteSession(SessionId id);

  /// <summary>
  /// Thread-safe mutation path. Applies <paramref name="mutator"/> to the session and
  /// fires <see cref="ObserveSession"/> observers with the updated value.
  /// </summary>
  void UpdateSession(SessionId id, Action<TSession> mutator);
}
