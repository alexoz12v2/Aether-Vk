using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using System.Threading;

namespace AetherVk.Logic.Services;

/// <summary>
/// Concrete implementation of <see cref="ITabStateService{TSession}"/>.
/// One singleton is registered per distinct TSession type via DI.
/// </summary>
public sealed class TabStateService<TSession> : ITabStateService<TSession>
  where TSession : class, ITabSession, new()
{
  private readonly ISchedulerProvider _schedulers;
  private readonly bool               _isExclusive;

  // Monotonic counter — never decremented, never reused within an application lifetime.
  private int _nextNumber = 0;

  private readonly ConcurrentDictionary<SessionId, BehaviorSubject<TSession>> _sessions = new();
  private readonly BehaviorSubject<IReadOnlyList<SessionId>>                   _sessionListSubject;

  // ── Public contract ────────────────────────────────────────────────────────

  public bool IsExclusive => _isExclusive;

  public IReadOnlyList<SessionId> ActiveSessionIds => _sessionListSubject.Value;

  public IObservable<IReadOnlyList<SessionId>> ObserveSessionList() =>
    _sessionListSubject.AsObservable().ObserveOn(_schedulers.MainThread);

  public IObservable<TSession> ObserveSession(SessionId id)
  {
    if (!_sessions.TryGetValue(id, out var subject))
      throw new KeyNotFoundException($"Session {id} not found in {typeof(TSession).Name}.");
    return subject.AsObservable().ObserveOn(_schedulers.MainThread);
  }

  public TSession GetSession(SessionId id)
  {
    if (!_sessions.TryGetValue(id, out var subject))
      throw new KeyNotFoundException($"Session {id} not found in {typeof(TSession).Name}.");
    return subject.Value;
  }

  public SessionId CreateSession()
  {
    if (_isExclusive && _sessions.Count > 0)
      throw new InvalidOperationException(
        $"{typeof(TSession).Name} is marked [ExclusiveSession]: only one session may exist.");

    int number = Interlocked.Increment(ref _nextNumber);
    var id     = new SessionId(typeof(TSession), number);
    _sessions[id] = new BehaviorSubject<TSession>(new TSession());
    PublishList();
    return id;
  }

  public void DeleteSession(SessionId id)
  {
    // Exclusive-session types always have exactly one session — deletion is not permitted.
    if (_isExclusive)
      return;

    // Guard: ensure at least one session always survives.
    // Create the replacement before removing the old one so subscribers always
    // have a valid target to switch to.
    if (_sessions.Count == 1)
      CreateSession();

    if (_sessions.TryRemove(id, out var subject))
    {
      subject.OnCompleted();
      subject.Dispose();
    }
    PublishList();
  }

  public void UpdateSession(SessionId id, Action<TSession> mutator)
  {
    if (!_sessions.TryGetValue(id, out var subject))
      throw new KeyNotFoundException($"Session {id} not found in {typeof(TSession).Name}.");

    mutator(subject.Value);
    subject.OnNext(subject.Value);
  }

  // ── Construction ───────────────────────────────────────────────────────────

  public TabStateService(ISchedulerProvider schedulers)
  {
    _schedulers  = schedulers;
    _isExclusive = typeof(TSession)
      .GetCustomAttributes(typeof(ExclusiveSessionAttribute), false).Length > 0;

    _sessionListSubject = new BehaviorSubject<IReadOnlyList<SessionId>>(
      Array.Empty<SessionId>());

    // Eagerly create the first (and possibly only) session so that VMs
    // resolved from DI always find at least one session ready.
    CreateSession();
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  private void PublishList()
  {
    var snapshot = _sessions.Keys.OrderBy(s => s.Number).ToList().AsReadOnly();
    _sessionListSubject.OnNext(snapshot);
  }
}
