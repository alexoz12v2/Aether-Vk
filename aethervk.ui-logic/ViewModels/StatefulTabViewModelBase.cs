using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Disposables;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// Base class for all tab view models that participate in the session system.
/// <list type="bullet">
///   <item>Subscribes to <see cref="ITabStateService{TSession}"/> observables on construction.</item>
///   <item>Exposes session-switching commands consumed by <c>CommonTabHeader</c>.</item>
///   <item>Implements <see cref="IStatefulTabHeader"/> so the centralized header in
///     <c>TabGroupNodeView</c> can bind to it without knowing the concrete type.</item>
///   <item>Implements <see cref="IDisposable"/> — call <c>Dispose()</c> when the last tab
///     referencing this VM is closed (handled by <c>TabGroupNodeViewModel.CloseTab</c>).</item>
/// </list>
/// </summary>
public abstract partial class StatefulTabViewModelBase<TSession>
  : TabItemViewModel,
    IStatefulTabHeader,
    IDisposable
  where TSession : class, ITabSession, new()
{
  private readonly ITabStateService<TSession> _sessionService;

  // All Rx subscriptions that should be torn down when this VM is disposed.
  private readonly CompositeDisposable _allSubs  = new();
  // Only the subscription to the *current* session's data stream.
  // Replaced atomically when the user switches sessions.
  private readonly SerialDisposable    _dataSub  = new();

  // ── Observable properties (IStatefulTabHeader + bindings) ─────────────────

  [ObservableProperty]
  private SessionId _sessionId;

  [ObservableProperty]
  private TSession? _currentSession;

  [ObservableProperty]
  private IReadOnlyList<SessionId> _availableSessions = Array.Empty<SessionId>();

  [ObservableProperty]
  private bool _isExclusiveSession;

  // ── Construction ───────────────────────────────────────────────────────────

  protected StatefulTabViewModelBase(
    string                      title,
    ITabStateService<TSession>  sessionService)
    : base(title)
  {
    _sessionService    = sessionService;
    IsExclusiveSession = sessionService.IsExclusive;

    // SerialDisposable: disposing it also disposes whatever is currently assigned.
    _allSubs.Add(_dataSub);

    // Boot into the first (and for exclusive types, the only) available session.
    _sessionId = sessionService.ActiveSessionIds[0];

    // Keep the session list in sync.
    sessionService.ObserveSessionList()
      .Subscribe(ids => AvailableSessions = ids)
      .AddDisposableTo(_allSubs);

    // Subscribe to the initial session's data stream.
    SubscribeToSession(_sessionId);
  }

  protected StatefulTabViewModelBase(
    string                      title,
    ITabStateService<TSession>  sessionService,
    IMessenger                  messenger)
    : base(title, messenger)
  {
    _sessionService    = sessionService;
    IsExclusiveSession = sessionService.IsExclusive;

    _allSubs.Add(_dataSub);

    _sessionId = sessionService.ActiveSessionIds[0];

    sessionService.ObserveSessionList()
      .Subscribe(ids => AvailableSessions = ids)
      .AddDisposableTo(_allSubs);

    SubscribeToSession(_sessionId);
  }

  // ── Session management commands ────────────────────────────────────────────

  [RelayCommand]
  private void SwitchSession(int number)
  {
    var target = AvailableSessions.FirstOrDefault(s => s.Number == number);
    if (target == default || target == SessionId)
      return;

    SubscribeToSession(target);
    SessionId = target;
  }

  [RelayCommand]
  private void DeleteSession(int number)
  {
    var target = AvailableSessions.FirstOrDefault(s => s.Number == number);
    if (target == default)
      return;

    _sessionService.DeleteSession(target);

    // If we just deleted our own session, switch to whatever the service left us with.
    if (target == SessionId)
    {
      var fallback = AvailableSessions.FirstOrDefault();
      if (fallback != default)
      {
        SubscribeToSession(fallback);
        SessionId = fallback;
      }
    }
  }

  [RelayCommand]
  private void NewSession()
  {
    // Safety guard — the UI hides this button for exclusive tabs.
    if (_sessionService.IsExclusive)
      return;

    var id = _sessionService.CreateSession();
    SubscribeToSession(id);
    SessionId = id;
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  private void SubscribeToSession(SessionId id)
  {
    // SerialDisposable automatically disposes the previous subscription before assigning the new one.
    _dataSub.Disposable = _sessionService
      .ObserveSession(id)
      .Subscribe(s => CurrentSession = s);
  }

  // ── IDisposable ────────────────────────────────────────────────────────────

  public virtual void Dispose()
  {
    IsActive = false;   // → OnDeactivated() → Messenger.UnregisterAll(this)
    _allSubs.Dispose(); // also disposes _dataSub and its current inner subscription
  }
}
