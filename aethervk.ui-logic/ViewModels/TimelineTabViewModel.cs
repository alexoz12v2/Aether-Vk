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
  private readonly INativeRuntimeService _runtimeService;
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

  public System.Collections.Generic.IReadOnlyList<SimulationSpeed> AvailableSpeeds { get; } = new[]
  {
      SimulationSpeed.OneHourPerSec,
      SimulationSpeed.ThreeHoursPerSec,
      SimulationSpeed.OneDayPerSec
  };

  public enum SimulationSpeed : int
  {
    OneHourPerSec = 2,
    ThreeHoursPerSec = 3,
    OneDayPerSec = 4
  }

  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(PlayPauseCommand))]
  private SimulationSpeed? _selectedSpeed;

  private bool CanPlayPause() => SelectedSpeed.HasValue;

  partial void OnIsPlayingChanged(bool value)
  {
    if (value && SelectedSpeed.HasValue)
      _timelineService.Play((int)SelectedSpeed.Value);
    else
      _timelineService.Pause();
  }

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

  public IRelayCommand RestoreCommand { get; }
  public IRelayCommand PlayPauseCommand { get; }
  public ICommand ResetCommand { get; }
  public ICommand RunToEndCommand { get; }

  public TimelineTabViewModel(
    ITranslationService translationService,
    ISchedulerProvider schedulerProvider,
    ITabStateService<TimelineSession> sessionService,
    ITabStateService<CometSession> cometSessionService,
    TimelineService timelineService,
    INativeRuntimeService runtimeService,
    ICometMessenger cometMessenger)
    : base("Timeline", sessionService, cometMessenger)
  {
    _translationService = translationService;
    _timelineSessionService = sessionService;
    _cometSessionService = cometSessionService;
    _timelineService = timelineService;
    _runtimeService = runtimeService;
    Icon = "⏱"; // stopwatch — U+23F1

    RestoreCommand = new RelayCommand(Restore);
    PlayPauseCommand = new RelayCommand(() => IsPlaying = !IsPlaying, CanPlayPause);
    ResetCommand = new RelayCommand(() => { IsPlaying = false; _timelineService.Reset(); });
    RunToEndCommand = new RelayCommand(() => { IsPlaying = false; });

    _timelineService.IsTimelineValid
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(isValid => IsTimelineValid = isValid)
      .AddDisposableTo(_disposables);

    Observable.Interval(TimeSpan.FromMilliseconds(33), schedulerProvider.MainThread)
      .Where(_ => IsPlaying)
      .Subscribe(_ => PollSimulationClock())
      .AddDisposableTo(_disposables);

    SubscribeToStrings(schedulerProvider);
    Restore();
    
    IsActive = true;
  }

  private void PollSimulationClock()
  {
    if (_runtimeService.GetSimulationClock(out var cent, out var ns))
    {
      string startEpochStr = CurrentSession?.CommittedStartEpoch ?? "";
      string endEpochStr = CurrentSession?.CommittedEndEpoch ?? "";

      if (string.IsNullOrEmpty(startEpochStr) || string.IsNullOrEmpty(endEpochStr))
      {
         return;
      }

      var currentDt = new DateTimeOffset(2000, 1, 1, 12, 0, 0, TimeSpan.Zero)
         .AddTicks((long)cent * TimeSpan.TicksPerDay * 36525L + (long)(ns / 100UL));
      
      var currentStr = currentDt.ToString("yyyy-MM-dd HH:mm");

      if (TimeUtils.TryParseIso8601(startEpochStr, out var startDt) && TimeUtils.TryParseIso8601(endEpochStr, out var endDt))
      {
         var startTai = TimeUtils.ToTaiParts(startDt);
         var endTai = TimeUtils.ToTaiParts(endDt);

         double startTicks = (double)startTai.centuries * TimeSpan.TicksPerDay * 36525.0 * 1e7 + startTai.nanoseconds / 100.0;
         double endTicks = (double)endTai.centuries * TimeSpan.TicksPerDay * 36525.0 * 1e7 + endTai.nanoseconds / 100.0;
         double currentTicks = (double)cent * TimeSpan.TicksPerDay * 36525.0 * 1e7 + ns / 100.0;

         double progress = (currentTicks - startTicks) / (endTicks - startTicks) * 100.0;
         Progress = Math.Max(0, Math.Min(100, progress));

         // Format the start epoch the same way to compare
         var startFormatted = startDt.ToString("yyyy-MM-dd HH:mm");
         if (currentStr != startFormatted)
         {
             // Publish the current epoch string to the session for Overlay window
             _timelineSessionService.UpdateSession(SessionId, s =>
             {
                 s.CurrentEpochString = currentStr;
             });
         }
         else
         {
             _timelineSessionService.UpdateSession(SessionId, s =>
             {
                 s.CurrentEpochString = string.Empty;
             });
         }
      }
    }
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
