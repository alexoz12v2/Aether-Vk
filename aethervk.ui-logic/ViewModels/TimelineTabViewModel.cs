using System;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Windows.Input;
using AetherVk.Logic.Attributes;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

[GenerateLocalizedStrings(
  keyPrefix: "Tabs_Timeline_",
  designTitle: "Timeline",
  designIcon: "⏱")]
public partial class TimelineTabViewModel : StatefulTabViewModelBase<TimelineSession>, ITimelineTabViewModel
{
  private readonly ITranslationService _translationService;
  private readonly TimelineService _timelineService;
  private readonly ITabStateService<TimelineSession> _timelineSessionService;
  private readonly ITabStateService<CometSession> _cometSessionService;
  private readonly CompositeDisposable _disposables = [];

  [ObservableProperty]
  private string _startEpoch = string.Empty;
  partial void OnStartEpochChanged(string value) => CheckUncommittedChanges();

  [ObservableProperty]
  private string _endEpoch = string.Empty;
  partial void OnEndEpochChanged(string value) => CheckUncommittedChanges();

  [ObservableProperty]
  private bool _hasUncommittedChanges;

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

  protected override void OnPropertyChanged(System.ComponentModel.PropertyChangedEventArgs e)
  {
    base.OnPropertyChanged(e);
    if (e.PropertyName == nameof(CurrentSession))
    {
      Restore();
      RefreshCommittedState();
    }
  }

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

    SubmitCommand = new RelayCommand(Submit);
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
  }

  private void RefreshCommittedState()
  {
    HasCommittedState = CurrentSession != null
      && !string.IsNullOrEmpty(CurrentSession.CommittedStartEpoch)
      && !string.IsNullOrEmpty(CurrentSession.CommittedEndEpoch);
  }

  private void CheckUncommittedChanges()
  {
    if (CurrentSession == null) return;
    HasUncommittedChanges = StartEpoch != CurrentSession.CommittedStartEpoch
                         || EndEpoch   != CurrentSession.CommittedEndEpoch;
    HasError = false;
    ErrorMessage = string.Empty;
  }

  private void Restore()
  {
    if (CurrentSession == null) return;
    StartEpoch = CurrentSession.CommittedStartEpoch;
    EndEpoch   = CurrentSession.CommittedEndEpoch;
    HasError = false;
    ErrorMessage = string.Empty;
    RefreshCommittedState();
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
    var cometSession = _cometSessionService.GetSession(cometSessionId);
    if (cometSession == null || cometSession.SpkId == null)
    {
      HasError = true;
      ErrorMessage = "A comet must be selected first.";
      return;
    }

    var startTai = TimeUtils.ToTaiParts(startDt);
    var endTai = TimeUtils.ToTaiParts(endDt);
    var range = new TimeRange(startTai.centuries, startTai.nanoseconds, endTai.centuries, endTai.nanoseconds);

    if (!_timelineService.CheckAlmanacCoverage(cometSession.SpkId.Value, range))
    {
      HasError = true;
      ErrorMessage = "The timeline is not fully covered by the selected comet's almanac.";
      return;
    }

    if (_timelineService.RequestEpochRange(range))
    {
      _timelineSessionService.UpdateSession(SessionId, s =>
      {
        s.CommittedStartEpoch = StartEpoch;
        s.CommittedEndEpoch   = EndEpoch;
      });
      RefreshCommittedState();
      CheckUncommittedChanges();
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
