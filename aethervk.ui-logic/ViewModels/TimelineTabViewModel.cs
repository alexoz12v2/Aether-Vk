using System;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Windows.Input;
using AetherVk.Logic.Attributes;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

using CommunityToolkit.Mvvm.Messaging;
using AetherVk.Logic.Messages;

namespace AetherVk.Logic.ViewModels;

[GenerateLocalizedStrings(
  keyPrefix: "Tabs_Timeline_",
  designTitle: "Timeline",
  designIcon: "⏱")]
public partial class TimelineTabViewModel : StatefulTabViewModelBase<TimelineSession>, ITimelineTabViewModel, IRecipient<CometDecommittedMessage>, IRecipient<CometCommittedMessage>
{
  private readonly ITranslationService _translationService;
  private readonly TimelineService _timelineService;
  private readonly ITabStateService<TimelineSession> _timelineSessionService;
  private readonly ITabStateService<CometSession> _cometSessionService;
  private readonly CompositeDisposable _disposables = [];

  [ObservableProperty]
  private string _startEpoch = string.Empty;
  partial void OnStartEpochChanged(string value)
  {
    CheckProposedVsCommitted();
    TryPropose();
  }

  [ObservableProperty]
  private string _endEpoch = string.Empty;
  partial void OnEndEpochChanged(string value)
  {
    CheckProposedVsCommitted();
    TryPropose();
  }

  /// <summary>
  /// True when the values currently in the text boxes differ from the last committed range.
  /// This means the user has typed (or a Propose was restored) something that hasn't been
  /// committed to the runtime yet.
  /// </summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(CanRestore))]
  private bool _isProposedDifferentFromCommitted;

  public bool CanRestore => HasCommittedState && IsProposedDifferentFromCommitted;

  [ObservableProperty]
  private bool _hasError;

  [ObservableProperty]
  private string _errorMessage = string.Empty;

  [ObservableProperty]
  private bool _isTimelineValid;

  [ObservableProperty]
  private bool _isPlaying;

  /// <summary>
  /// Playback progress in the range [0, 100]. Driven externally once the
  /// simulation clock is wired; starts at 0.
  /// </summary>
  [ObservableProperty]
  private double _progress;

  /// <summary>
  /// True only when the current session has a confirmed committed epoch pair.
  /// Controls visibility of the playback toolbar and progress bar.
  /// </summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(CanRestore))]
  private bool _hasCommittedState;
  
  [ObservableProperty]
  private bool _hasProposedState;

  [ObservableProperty]
  private string _displayProposedRange = string.Empty;

  [ObservableProperty]
  private string _displayCommittedRange = string.Empty;

  protected override void OnPropertyChanged(System.ComponentModel.PropertyChangedEventArgs e)
  {
    base.OnPropertyChanged(e);
    if (e.PropertyName == nameof(CurrentSession))
    {
      Restore();
      RefreshCommittedState();
      RefreshProposedState();
    }
  }

  public ICommand RestoreCommand { get; }
  public ICommand PlayPauseCommand { get; }
  public ICommand ResetCommand { get; }
  public ICommand RunToEndCommand { get; }

  public TimelineTabViewModel(
    ITranslationService translationService,
    ISchedulerProvider schedulerProvider,
    ITabStateService<TimelineSession> sessionService,
    ITabStateService<CometSession> cometSessionService,
    TimelineService timelineService,
    ICometMessenger cometMessenger)
    : base("Timeline", sessionService, cometMessenger)
  {
    _translationService = translationService;
    _timelineSessionService = sessionService;
    _cometSessionService = cometSessionService;
    _timelineService = timelineService;
    Icon = "⏱"; // stopwatch — U+23F1

    RestoreCommand = new RelayCommand(Restore);
    PlayPauseCommand = new RelayCommand(() => IsPlaying = !IsPlaying);
    ResetCommand = new RelayCommand(() => { IsPlaying = false; });
    RunToEndCommand = new RelayCommand(() => { IsPlaying = false; });

    _timelineService.IsTimelineValid
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(isValid => IsTimelineValid = isValid)
      .AddDisposableTo(_disposables);

    SubscribeToStrings(schedulerProvider);
    Restore();
    IsActive = true;  // → OnActivated() → registers CometCommitted/CometDecommitted
  }

  protected override void OnActivated()
  {
    Messenger.Register<TimelineTabViewModel, CometCommittedMessage>(this, (r, m) => r.Receive(m));
    Messenger.Register<TimelineTabViewModel, CometDecommittedMessage>(this, (r, m) => r.Receive(m));
  }

  public void Receive(CometDecommittedMessage message)
  {
    if (CurrentSession == null) return;

    // Per spec: discard committed → flow its value back to proposed first.
    var prevStart = CurrentSession.CommittedStartEpoch;
    var prevEnd   = CurrentSession.CommittedEndEpoch;

    _timelineSessionService.UpdateSession(SessionId, s =>
    {
      // Only flow back if there was actually a committed range.
      if (!string.IsNullOrEmpty(s.CommittedStartEpoch))
      {
        s.ProposedStartEpoch = s.CommittedStartEpoch;
        s.ProposedEndEpoch   = s.CommittedEndEpoch;
      }
      s.CommittedStartEpoch = string.Empty;
      s.CommittedEndEpoch   = string.Empty;
    });

    // Re-push the now-restored proposed range to the service so
    // CometTabViewModel (and any other subscriber) sees the update.
    if (!string.IsNullOrEmpty(prevStart) && !string.IsNullOrEmpty(prevEnd)
        && TimeUtils.TryParseIso8601(prevStart, out var startDt)
        && TimeUtils.TryParseIso8601(prevEnd,   out var endDt))
    {
      var s2 = TimeUtils.ToTaiParts(startDt);
      var e2 = TimeUtils.ToTaiParts(endDt);
      _timelineService.ProposeEpochRange(
        new TimeRange(s2.centuries, s2.nanoseconds, e2.centuries, e2.nanoseconds));
    }

    // Refresh UI to reflect cleared committed and (possibly) updated proposed.
    StartEpoch = CurrentSession.ProposedStartEpoch;
    EndEpoch   = CurrentSession.ProposedEndEpoch;
    RefreshCommittedState();
    RefreshProposedState();
    CheckProposedVsCommitted();
  }

  private void RefreshCommittedState()
  {
    HasCommittedState = CurrentSession != null
      && !string.IsNullOrEmpty(CurrentSession.CommittedStartEpoch)
      && !string.IsNullOrEmpty(CurrentSession.CommittedEndEpoch);
    if (HasCommittedState && CurrentSession != null)
    {
      DisplayCommittedRange = $"{CurrentSession.CommittedStartEpoch} to {CurrentSession.CommittedEndEpoch}";
    }
  }

  private void RefreshProposedState()
  {
    HasProposedState = CurrentSession != null
      && !string.IsNullOrEmpty(CurrentSession.ProposedStartEpoch)
      && !string.IsNullOrEmpty(CurrentSession.ProposedEndEpoch);
    if (HasProposedState && CurrentSession != null)
    {
      DisplayProposedRange = $"{CurrentSession.ProposedStartEpoch} to {CurrentSession.ProposedEndEpoch}";
    }
  }

  private void CheckProposedVsCommitted()
  {
    if (CurrentSession == null) return;
    // The text boxes always carry the proposed range. The indicator fires
    // when the proposed range diverges from the last committed range.
    IsProposedDifferentFromCommitted =
      StartEpoch != CurrentSession.CommittedStartEpoch ||
      EndEpoch   != CurrentSession.CommittedEndEpoch;
  }

  private void Restore()
  {
    if (CurrentSession == null) return;

    // Seed default proposed range if the session has never had one set.
    // TODO: read defaults from a configuration file in the future.
    bool needsSeed = string.IsNullOrEmpty(CurrentSession.ProposedStartEpoch)
                  || string.IsNullOrEmpty(CurrentSession.ProposedEndEpoch);
    if (needsSeed)
    {
      _timelineSessionService.UpdateSession(SessionId, s =>
      {
        if (string.IsNullOrEmpty(s.ProposedStartEpoch))
          s.ProposedStartEpoch = "2025-10-01T00:00:00Z";
        if (string.IsNullOrEmpty(s.ProposedEndEpoch))
          s.ProposedEndEpoch = "2025-11-11T00:00:00Z";
      });
    }

    StartEpoch = CurrentSession.ProposedStartEpoch;
    EndEpoch   = CurrentSession.ProposedEndEpoch;

    if (TimeUtils.TryParseIso8601(StartEpoch, out var startDt)
        && TimeUtils.TryParseIso8601(EndEpoch,   out var endDt))
    {
      var startTai = TimeUtils.ToTaiParts(startDt);
      var endTai   = TimeUtils.ToTaiParts(endDt);
      _timelineService.ProposeEpochRange(
        new TimeRange(startTai.centuries, startTai.nanoseconds,
                      endTai.centuries,   endTai.nanoseconds));
    }

    HasError = false;
    ErrorMessage = string.Empty;
    RefreshCommittedState();
    RefreshProposedState();
    CheckProposedVsCommitted();
  }

  private bool _isProposing;
  private void TryPropose()
  {
    if (_isProposing) return;
    _isProposing = true;
    try
    {
      HasError = false;
      ErrorMessage = string.Empty;
      if (CurrentSession == null) return;

      if (!TimeUtils.TryParseIso8601(StartEpoch, out var startDt) ||
          !TimeUtils.TryParseIso8601(EndEpoch, out var endDt))
      {
        StartEpoch = CurrentSession.ProposedStartEpoch;
        EndEpoch = CurrentSession.ProposedEndEpoch;
        return;
      }

      var diff = endDt - startDt;
      if (diff.TotalDays < 0 || diff.TotalDays < 28)
      {
        StartEpoch = CurrentSession.ProposedStartEpoch;
        EndEpoch = CurrentSession.ProposedEndEpoch;
        return;
      }

      var startTai = TimeUtils.ToTaiParts(startDt);
      var endTai   = TimeUtils.ToTaiParts(endDt);
      var range    = new TimeRange(startTai.centuries, startTai.nanoseconds,
                                   endTai.centuries,   endTai.nanoseconds);
      PersistProposal(StartEpoch, EndEpoch, range);
    }
    finally
    {
      _isProposing = false;
    }
  }

  /// <summary>
  /// Persists a validated proposed range to both the session store and the
  /// reactive service without performing any user-facing validation.
  /// Use this from internal paths (Restore, Submit success/rollback) instead of
  /// calling the public <see cref="Propose"/> command so that validation side-effects
  /// (clearing errors, etc.) are not inadvertently triggered.
  /// </summary>
  private void PersistProposal(string start, string end, TimeRange range)
  {
    _timelineSessionService.UpdateSession(SessionId, s =>
    {
      s.ProposedStartEpoch = start;
      s.ProposedEndEpoch   = end;
    });
    _timelineService.ProposeEpochRange(range);
    RefreshProposedState();
    CheckProposedVsCommitted();
  }

  public void Receive(CometCommittedMessage message)
  {
    if (CurrentSession == null) return;

    _timelineSessionService.UpdateSession(SessionId, s =>
    {
      s.CommittedStartEpoch = StartEpoch;
      s.CommittedEndEpoch   = EndEpoch;
    });

    RefreshCommittedState();
    CheckProposedVsCommitted();
  }

  private void SubscribeToStrings(ISchedulerProvider schedulerProvider)
  {
    RefreshStrings();
    _translationService.CultureChanged
      .Skip(1)
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(_ => RefreshStrings())
      .AddDisposableTo(_disposables);
  }
}
