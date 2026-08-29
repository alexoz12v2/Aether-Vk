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
public partial class TimelineTabViewModel : StatefulTabViewModelBase<TimelineSession>, ITimelineTabViewModel, IRecipient<CometDecommittedMessage>
{
  private readonly ITranslationService _translationService;
  private readonly TimelineService _timelineService;
  private readonly ITabStateService<TimelineSession> _timelineSessionService;
  private readonly ITabStateService<CometSession> _cometSessionService;
  private readonly CompositeDisposable _disposables = [];

  [ObservableProperty]
  private string _startEpoch = string.Empty;
  partial void OnStartEpochChanged(string value) => CheckProposedVsCommitted();

  [ObservableProperty]
  private string _endEpoch = string.Empty;
  partial void OnEndEpochChanged(string value) => CheckProposedVsCommitted();

  /// <summary>
  /// True when the values currently in the text boxes differ from the last committed range.
  /// This means the user has typed (or a Propose was restored) something that hasn't been
  /// committed to the runtime yet.
  /// </summary>
  [ObservableProperty]
  private bool _isProposedDifferentFromCommitted;

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

  public ICommand ProposeCommand { get; }
  public ICommand SubmitCommand { get; }
  public ICommand RestoreCommand { get; }
  public ICommand PlayPauseCommand { get; }
  public ICommand ResetCommand { get; }
  public ICommand RunToEndCommand { get; }

  public TimelineTabViewModel(
    ITranslationService translationService,
    ISchedulerProvider schedulerProvider,
    ITabStateService<TimelineSession> sessionService,
    ITabStateService<CometSession> cometSessionService,
    TimelineService timelineService)
    : base("Timeline", sessionService)
  {
    _translationService = translationService;
    _timelineSessionService = sessionService;
    _cometSessionService = cometSessionService;
    _timelineService = timelineService;
    Icon = "⏱"; // stopwatch — U+23F1

    ProposeCommand = new RelayCommand(Propose);
    SubmitCommand = new RelayCommand(Submit);
    RestoreCommand = new RelayCommand(Restore);
    PlayPauseCommand = new RelayCommand(() => IsPlaying = !IsPlaying);
    ResetCommand = new RelayCommand(() => { IsPlaying = false; });
    RunToEndCommand = new RelayCommand(() => { IsPlaying = false; });

    _timelineService.IsTimelineValid
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(isValid => IsTimelineValid = isValid)
      .AddDisposableTo(_disposables);

    WeakReferenceMessenger.Default.Register(this);

    SubscribeToStrings(schedulerProvider);
    Restore();
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

  private void Propose()
  {
    HasError = false;
    ErrorMessage = string.Empty;
    if (CurrentSession == null) return;

    if (!TimeUtils.TryParseIso8601(StartEpoch, out var startDt))
    {
      HasError = true;
      ErrorMessage = "Invalid Start Epoch format.";
      return;
    }

    if (!TimeUtils.TryParseIso8601(EndEpoch, out var endDt))
    {
      HasError = true;
      ErrorMessage = "Invalid End Epoch format.";
      return;
    }

    var diff = endDt - startDt;
    if (diff.TotalDays < 0)
    {
      HasError = true;
      ErrorMessage = "Reverse time ranges are not allowed.";
      return;
    }
    if (diff.TotalDays < 28)
    {
      HasError = true;
      ErrorMessage = "Date range must be at least one month.";
      return;
    }

    var startTai = TimeUtils.ToTaiParts(startDt);
    var endTai   = TimeUtils.ToTaiParts(endDt);
    var range    = new TimeRange(startTai.centuries, startTai.nanoseconds,
                                 endTai.centuries,   endTai.nanoseconds);
    PersistProposal(StartEpoch, EndEpoch, range);
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

  private void Submit()
  {
    HasError = false;
    ErrorMessage = string.Empty;
    if (CurrentSession == null) return;

    if (!TimeUtils.TryParseIso8601(StartEpoch, out var startDt))
    {
      HasError = true;
      ErrorMessage = "Invalid Start Epoch format.";
      return;
    }

    if (!TimeUtils.TryParseIso8601(EndEpoch, out var endDt))
    {
      HasError = true;
      ErrorMessage = "Invalid End Epoch format.";
      return;
    }

    var diff = endDt - startDt;
    if (diff.TotalDays < 0)
    {
      HasError = true;
      ErrorMessage = "Reverse time ranges are not allowed.";
      return;
    }
    if (diff.TotalDays < 28) // Using 28 days as "less than a month"
    {
      HasError = true;
      ErrorMessage = "Date range must be at least one month.";
      return;
    }

    var cometSessionId = _cometSessionService.ActiveSessionIds[0];
    var cometSession   = _cometSessionService.GetSession(cometSessionId);
    if (cometSession == null || cometSession.SpkId == null)
    {
      HasError = true;
      ErrorMessage = "A comet must be selected first.";
      return;
    }

    var startTai = TimeUtils.ToTaiParts(startDt);
    var endTai   = TimeUtils.ToTaiParts(endDt);
    var range    = new TimeRange(startTai.centuries, startTai.nanoseconds,
                                 endTai.centuries,   endTai.nanoseconds);

    if (!_timelineService.CheckAlmanacCoverage(cometSession.SpkId.Value, range))
    {
      HasError = true;
      ErrorMessage = "The timeline is not fully covered by the selected comet's almanac. Restoring proposed from committed.";

      // Rollback: proposed snaps back to the last committed range if one exists.
      if (!string.IsNullOrEmpty(CurrentSession.CommittedStartEpoch)
          && TimeUtils.TryParseIso8601(CurrentSession.CommittedStartEpoch, out var cs)
          && TimeUtils.TryParseIso8601(CurrentSession.CommittedEndEpoch,   out var ce))
      {
        StartEpoch = CurrentSession.CommittedStartEpoch;
        EndEpoch   = CurrentSession.CommittedEndEpoch;
        var csTai = TimeUtils.ToTaiParts(cs);
        var ceTai = TimeUtils.ToTaiParts(ce);
        PersistProposal(StartEpoch, EndEpoch,
          new TimeRange(csTai.centuries, csTai.nanoseconds, ceTai.centuries, ceTai.nanoseconds));
      }
      return;
    }

    if (_timelineService.RequestEpochRange(range))
    {
      _timelineSessionService.UpdateSession(SessionId, s =>
      {
        s.CommittedStartEpoch = StartEpoch;
        s.CommittedEndEpoch   = EndEpoch;
      });
      // Keep proposed in sync with what was just committed.
      PersistProposal(StartEpoch, EndEpoch, range);
      RefreshCommittedState();
    }
    else
    {
      HasError = true;
      ErrorMessage = "Failed to submit timeline to runtime.";
    }
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
