using System;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Threading.Tasks;
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
  [NotifyPropertyChangedFor(nameof(IsRangeLocked))]
  private bool _isProposedDifferentFromCommitted;

  public bool CanRestore => HasCommittedState && IsProposedDifferentFromCommitted;

  /// <summary>
  /// The boxes were edited after a commit: the engine keeps the committed range (reviewer report
  /// 2026-10-29, "it picks other dates") until the comet is committed again with the new one.
  /// </summary>
  public bool IsRangeLocked => HasCommittedState && IsProposedDifferentFromCommitted;

  public const string RangeLockedText =
    "Range locked while the comet is committed: the simulation keeps the committed range. Decommit to change it, or recommit with this range.";

  /// <summary>Commits the comet again with the boxes' range (the Comet tab's Download &amp; Commit).</summary>
  [RelayCommand]
  private void RecommitWithRange()
  {
    if (IsRangeLocked && !HasError)
      _timelineService.RequestRecommit();
  }

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
      SimulationSpeed.TenSecondsPerSec,
      SimulationSpeed.OneMinutePerSec,
      SimulationSpeed.OneHourPerSec,
      SimulationSpeed.ThreeHoursPerSec,
      SimulationSpeed.OneDayPerSec
  };

  /// <summary>Wire values of <c>SimSpeed</c> (oshal time.rs): appended, never renumbered.</summary>
  public enum SimulationSpeed : int
  {
    OneHourPerSec = 2,
    ThreeHoursPerSec = 3,
    OneDayPerSec = 4,
    /// <summary>60×: watching dust emission windows near the nucleus.</summary>
    OneMinutePerSec = 5,
    /// <summary>10×</summary>
    TenSecondsPerSec = 6,
  }

  /// <summary>A speed preset with its localized label (ComboBox item).</summary>
  public sealed record SpeedOption(SimulationSpeed Speed, string Label);

  /// <summary><see cref="AvailableSpeeds"/> with localized labels; re-raised on culture change.</summary>
  public System.Collections.Generic.IReadOnlyList<SpeedOption> SpeedOptions =>
    System.Linq.Enumerable.ToArray(System.Linq.Enumerable.Select(AvailableSpeeds, s => new SpeedOption(s, SpeedLabel(s))));

  internal string SpeedLabel(SimulationSpeed s) => s switch
  {
    SimulationSpeed.TenSecondsPerSec => StrSpeed10s,
    SimulationSpeed.OneMinutePerSec => StrSpeed1min,
    SimulationSpeed.OneHourPerSec => StrSpeed1h,
    SimulationSpeed.ThreeHoursPerSec => StrSpeed3h,
    SimulationSpeed.OneDayPerSec => StrSpeed1d,
    _ => s.ToString(),
  };


  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(PlayPauseCommand))]
  private SimulationSpeed? _selectedSpeed;

  private bool CanPlayPause() => SelectedSpeed.HasValue;

  // the session keeps the speed for the savefile (TimelineDto.Speed)
  partial void OnSelectedSpeedChanged(SimulationSpeed? value) =>
    _timelineSessionService?.UpdateSession(SessionId, s => s.Speed = (int?)value);

  partial void OnIsPlayingChanged(bool value)
  {
    if (value && SelectedSpeed.HasValue)
    {
        _timelineSessionService.UpdateSession(SessionId, s =>
        {
            s.SnapshotStartEpoch = s.CommittedStartEpoch;
            s.SnapshotEndEpoch = s.CommittedEndEpoch;
        });
        _timelineService.Play((int)SelectedSpeed.Value);
    }
    else
    {
        _timelineService.Pause();
    }
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
  [NotifyPropertyChangedFor(nameof(IsRangeLocked))]
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
    // localized speed labels changed (culture switch): the ComboBox items follow
    if (e.PropertyName is { } n && n.StartsWith("StrSpeed", StringComparison.Ordinal))
      base.OnPropertyChanged(new System.ComponentModel.PropertyChangedEventArgs(nameof(SpeedOptions)));
    if (e.PropertyName == nameof(CurrentSession))
    {
      HookRestored(CurrentSession);
      Restore();
      RefreshCommittedState();
      RefreshProposedState();
    }
  }

  public IRelayCommand RestoreCommand { get; }
  public IRelayCommand PlayPauseCommand { get; }
  public ICommand ResetCommand { get; }
  public IAsyncRelayCommand RunToEndCommand { get; }
  public IAsyncRelayCommand SeekToEpochCommand { get; }

  /// <summary>Display and input format of the epoch box (the viewport's date, to the second).</summary>
  public const string EpochBoxFormat = "yyyy-MM-dd HH:mm:ss";

  /// <summary>
  /// The engine clock in <see cref="EpochBoxFormat"/>; the user types an epoch here and presses
  /// Enter (<see cref="SeekToEpochCommand"/>) to seek. Not refreshed while <see cref="IsEditingCurrentEpoch"/>.
  /// </summary>
  [ObservableProperty]
  private string _currentEpochText = string.Empty;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(HasCurrentEpochError))]
  private string _currentEpochError = string.Empty;

  public bool HasCurrentEpochError => CurrentEpochError.Length > 0;

  /// <summary>Set by the view while the box has focus, so playback does not overwrite typing.</summary>
  public bool IsEditingCurrentEpoch { get; set; }

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

    RestoreCommand = new RelayCommand(RestoreCommitted);
    PlayPauseCommand = new RelayCommand(() => IsPlaying = !IsPlaying, CanPlayPause);
    ResetCommand = new RelayCommand(() =>
    {
        IsPlaying = false;
        _timelineService.Reset();
        Progress = 0.0;
        CurrentEpochError = string.Empty;
        PollSimulationClock();
        
        // Clear current epoch in session so the 3D viewport overlay resets immediately
        _timelineSessionService.UpdateSession(SessionId, s =>
        {
            s.CurrentEpochString = string.Empty;
        });
    });

    _timelineService.IsSimulationRunning
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(running =>
      {
          if (!running && IsPlaying) IsPlaying = false; // external stop -> sync UI
      })
      .AddDisposableTo(_disposables);
    RunToEndCommand = new AsyncRelayCommand(RunToEndAsync);
    SeekToEpochCommand = new AsyncRelayCommand(SeekToEpochAsync);

    _timelineService.ModelRangeRequests
      .Subscribe(end =>
      {
        ModelEndEpoch = end.UtcDateTime.ToString("yyyy-MM-ddTHH:mm:ss", System.Globalization.CultureInfo.InvariantCulture);
        ApplyModelRange();
      })
      .AddDisposableTo(_disposables);

    // any seek (this tab, a FITS observation epoch): show where the clock now is
    _timelineService.Seeked
      .Subscribe(_ => PollSimulationClock())
      .AddDisposableTo(_disposables);

    // the range the engine actually holds (ExternalState::TimeRange): Run to End and the committed
    // labels follow it, never the textbox, so a refused or stale range cannot send the clock
    // past the user's end (towards the SPK's +30 d margin)
    _timelineService.ValidatedTimeRange
      .Subscribe(range => _confirmedRange = range)
      .AddDisposableTo(_disposables);

    _timelineService.IsTimelineValid
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(isValid => IsTimelineValid = isValid)
      .AddDisposableTo(_disposables);

    Observable.Interval(TimeSpan.FromMilliseconds(33), schedulerProvider.MainThread)
      .Where(_ => IsPlaying)
      .Subscribe(_ => PollSimulationClock())
      .AddDisposableTo(_disposables);

    SubscribeToStrings(schedulerProvider);
    HookRestored(CurrentSession);
    Restore();
    
    IsActive = true;
  }

  /// <summary>
  /// Run to End: pauses, then jumps to the committed end through the seek path (the clock is
  /// clamped to the end, bodies and the last TTL of dust rebuilt there; no frame is simulated
  /// on the way). The overlay spinner shows while the seek blocks off the UI thread; the
  /// timeline is left paused at the end.
  /// </summary>
  private async Task RunToEndAsync()
  {
    if (CurrentSession is null || RunToEndTarget(_confirmedRange, CurrentSession.CommittedEndEpoch) is not var (cent, ns))
      return;
    IsPlaying = false;
    _timelineSessionService.UpdateSession(SessionId, s => s.IsRunningToEnd = true);
    bool ok;
    try
    {
      ok = await _timelineService.SeekAsync(cent, ns);
    }
    finally
    {
      _timelineSessionService.UpdateSession(SessionId, s => s.IsRunningToEnd = false);
    }
    // even a timed-out seek moved the clock: show where it is
    PollSimulationClock();
    if (!ok) Console.WriteLine("[Timeline] Run to End: the seek did not confirm in time");
  }

  /// <summary>
  /// The epoch box: parses <see cref="CurrentEpochText"/> (UTC, as displayed), refuses it outside
  /// the committed range, else pauses and seeks there.
  /// </summary>
  private async Task SeekToEpochAsync()
  {
    if (CurrentSession is null || !HasCommittedState)
      return;
    if (!TryParseEpochBox(CurrentEpochText, out var target))
    {
      CurrentEpochError = $"Not an epoch: use {EpochBoxFormat}.";
      return;
    }
    if (EffectiveRange() is not { } range)
      return;
    if (!InRange(target, range))
    {
      CurrentEpochError = $"Outside the timeline ({FormatEpoch(range.StartCenturies, range.StartNs)} to {FormatEpoch(range.EndCenturies, range.EndNs)}).";
      return;
    }
    CurrentEpochError = string.Empty;
    IsPlaying = false;
    _timelineSessionService.UpdateSession(SessionId, s => s.IsRunningToEnd = true);
    bool ok;
    try
    {
      ok = await _timelineService.SeekAsync(target.Centuries, target.Nanoseconds);
    }
    finally
    {
      _timelineSessionService.UpdateSession(SessionId, s => s.IsRunningToEnd = false);
    }
    PollSimulationClock(force: true);
    if (!ok) Console.WriteLine("[Timeline] epoch box: the seek did not confirm in time");
  }

  /// <summary>The epoch box's text as an engine epoch: invariant culture, UTC unless an offset is typed.</summary>
  internal static bool TryParseEpochBox(string text, out (short Centuries, ulong Nanoseconds) epoch)
  {
    epoch = default;
    if (string.IsNullOrWhiteSpace(text)
        || !DateTimeOffset.TryParse(text.Trim(), System.Globalization.CultureInfo.InvariantCulture,
             System.Globalization.DateTimeStyles.AssumeUniversal, out var dt))
      return false;
    epoch = TimeUtils.ToTaiParts(dt);
    return true;
  }

  internal static string FormatEpoch(short cent, ulong ns) =>
    TimeUtils.FromTaiParts(cent, ns).ToString(EpochBoxFormat, System.Globalization.CultureInfo.InvariantCulture);

  internal static bool InRange((short Centuries, ulong Nanoseconds) e, TimeRange range) =>
    Compare(e, (range.StartCenturies, range.StartNs)) >= 0 && Compare(e, (range.EndCenturies, range.EndNs)) <= 0;

  private static int Compare((short C, ulong N) a, (short C, ulong N) b) =>
    a.C != b.C ? a.C.CompareTo(b.C) : a.N.CompareTo(b.N);

  /// <summary>The engine-confirmed range, else the committed text.</summary>
  private TimeRange? EffectiveRange()
  {
    if (_confirmedRange is not null)
      return _confirmedRange;
    if (CurrentSession is null
        || !TimeUtils.TryParseIso8601(CurrentSession.CommittedStartEpoch, out var startDt)
        || !TimeUtils.TryParseIso8601(CurrentSession.CommittedEndEpoch, out var endDt))
      return null;
    var a = TimeUtils.ToTaiParts(startDt);
    var b = TimeUtils.ToTaiParts(endDt);
    return new TimeRange(a.centuries, a.nanoseconds, b.centuries, b.nanoseconds);
  }

  /// <summary>
  /// Where Run to End seeks: exactly the engine-confirmed end (the user's end date as the engine
  /// holds it), else the committed end text; <c>null</c> without either.
  /// </summary>
  internal static (short Centuries, ulong Nanoseconds)? RunToEndTarget(TimeRange? confirmed, string committedEndEpoch)
  {
    if (confirmed is not null)
      return (confirmed.EndCenturies, confirmed.EndNs);
    if (TimeUtils.TryParseIso8601(committedEndEpoch, out var endDt))
      return TimeUtils.ToTaiParts(endDt);
    return null;
  }

  /// <summary>
  /// The viewport's date label for the engine clock (<paramref name="cent"/>, <paramref name="ns"/>):
  /// "yyyy-MM-dd HH:mm", or empty at the committed start (and without a committed start).
  /// </summary>
  internal static string ClockLabel(short cent, ulong ns, string committedStartEpoch)
  {
    if (!TimeUtils.TryParseIso8601(committedStartEpoch, out var startDt))
      return string.Empty;
    var currentDt = new DateTimeOffset(2000, 1, 1, 12, 0, 0, TimeSpan.Zero)
       .AddTicks((long)cent * TimeSpan.TicksPerDay * 36525L + (long)(ns / 100UL));
    var currentStr = currentDt.ToString("yyyy-MM-dd HH:mm");
    return currentStr != startDt.ToString("yyyy-MM-dd HH:mm") ? currentStr : string.Empty;
  }

  private void PollSimulationClock() => PollSimulationClock(force: false);

  /// <param name="force">also refresh the epoch box while it is being edited (after its own seek)</param>
  private void PollSimulationClock(bool force)
  {
    if (_runtimeService.GetSimulationClock(out var cent, out var ns))
    {
      if (force || !IsEditingCurrentEpoch)
        CurrentEpochText = FormatEpoch(cent, ns);
      string startEpochStr = CurrentSession?.CommittedStartEpoch ?? "";
      string endEpochStr = CurrentSession?.CommittedEndEpoch ?? "";

      if (string.IsNullOrEmpty(startEpochStr) || string.IsNullOrEmpty(endEpochStr))
      {
         return;
      }

      if (TimeUtils.TryParseIso8601(startEpochStr, out var startDt) && TimeUtils.TryParseIso8601(endEpochStr, out var endDt))
      {
         var startTai = TimeUtils.ToTaiParts(startDt);
         var endTai = TimeUtils.ToTaiParts(endDt);

         double startTicks = (double)startTai.centuries * TimeSpan.TicksPerDay * 36525.0 * 1e7 + startTai.nanoseconds / 100.0;
         double endTicks = (double)endTai.centuries * TimeSpan.TicksPerDay * 36525.0 * 1e7 + endTai.nanoseconds / 100.0;
         double currentTicks = (double)cent * TimeSpan.TicksPerDay * 36525.0 * 1e7 + ns / 100.0;

         double progress = (currentTicks - startTicks) / (endTicks - startTicks) * 100.0;
         Progress = Math.Max(0, Math.Min(100, progress));

         // Publish the current epoch string to the session for Overlay window
         var label = ClockLabel(cent, ns, startEpochStr);
         _timelineSessionService.UpdateSession(SessionId, s => s.CurrentEpochString = label);
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

  private TimelineSession? _restoredHooked;

  /// <summary>A savefile open replaces the session's values: re-read them (see <see cref="TimelineSession.Restored"/>).</summary>
  private void HookRestored(TimelineSession? session)
  {
    if (ReferenceEquals(_restoredHooked, session)) return;
    if (_restoredHooked is not null) _restoredHooked.Restored -= OnSessionRestored;
    if (session is not null) session.Restored += OnSessionRestored;
    _restoredHooked = session;
  }

  private void OnSessionRestored()
  {
    Restore();
    RefreshCommittedState();
    RefreshProposedState();
  }

  public override void Dispose()
  {
    HookRestored(null);
    _disposables.Dispose();
    base.Dispose();
  }

  /// <summary>The Restore button: the boxes back to the committed range (else the saved proposal).</summary>
  private void RestoreCommitted()
  {
    if (CurrentSession is { } s && HasCommittedState)
    {
      StartEpoch = s.CommittedStartEpoch;
      EndEpoch = s.CommittedEndEpoch;
      return;
    }
    Restore();
  }

  private void Restore()
  {
    if (CurrentSession == null || _timelineSessionService == null) return;

    // Seed the default proposed range if it was cleared.
    bool needsSeed = string.IsNullOrEmpty(CurrentSession.ProposedStartEpoch)
                  || string.IsNullOrEmpty(CurrentSession.ProposedEndEpoch);
    if (needsSeed)
    {
      _timelineSessionService.UpdateSession(SessionId, s =>
      {
        if (string.IsNullOrEmpty(s.ProposedStartEpoch))
          s.ProposedStartEpoch = TimelineSession.DefaultProposedStart;
        if (string.IsNullOrEmpty(s.ProposedEndEpoch))
          s.ProposedEndEpoch = TimelineSession.DefaultProposedEnd;
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

    // the saved speed (a savefile open); none saved keeps the current selection
    if (CurrentSession.Speed is int speed && Enum.IsDefined(typeof(SimulationSpeed), speed))
      SelectedSpeed = (SimulationSpeed)speed;

    HasError = false;
    ErrorMessage = string.Empty;
    RefreshCommittedState();
    RefreshProposedState();
    CheckProposedVsCommitted();
    // the engine's clock (a reopened save sits at its saved epoch, not at the start)
    if (!IsPlaying) PollSimulationClock();
  }

  private TimeRange? _confirmedRange;

  // ── "Model" preset (feedback #20): N rotations ending at the image epoch ─────────────────

  /// <summary>End of the model range (the image's epoch); empty: the end box.</summary>
  [ObservableProperty]
  private string _modelEndEpoch = string.Empty;

  /// <summary>Nucleus rotations the model range spans.</summary>
  [ObservableProperty]
  private int _modelRotations = 3;

  /// <summary>
  /// The range <paramref name="rotations"/> sidereal periods of <paramref name="periodHours"/>
  /// long ending at <paramref name="end"/>, at least <see cref="MinRangeDays"/>. Null for no
  /// rotation (period ≤ 0) or rotations &lt; 1.
  /// </summary>
  internal static (DateTimeOffset Start, DateTimeOffset End)? ModelRange(DateTimeOffset end, int rotations, double periodHours)
  {
    if (rotations < 1 || !(periodHours > 0) || double.IsInfinity(periodHours))
      return null;
    double hours = Math.Max(rotations * periodHours, MinRangeDays * 24.0);
    return (end - TimeSpan.FromHours(hours), end);
  }

  /// <summary>
  /// Proposes <see cref="ModelRotations"/> rotations ending at <see cref="ModelEndEpoch"/> (or the
  /// end box), the period from the committed comet's rotation rate. Commit as usual.
  /// </summary>
  [RelayCommand]
  private void ApplyModelRange()
  {
    string endText = string.IsNullOrWhiteSpace(ModelEndEpoch) ? EndEpoch : ModelEndEpoch;
    if (!TimeUtils.TryParseIso8601(endText, out var end))
    {
      SetProposalError("Model range: the end epoch is not a date (yyyy-MM-ddTHH:mm:ss).");
      return;
    }
    double rate = _cometSessionService.ActiveSessionIds.Count == 0
      ? 0
      : _cometSessionService.GetSession(_cometSessionService.ActiveSessionIds[0]).RotRateDegDay;
    double periodHours = rate == 0 ? 0 : 360.0 * 24.0 / Math.Abs(rate);
    if (ModelRange(end, ModelRotations, periodHours) is not { } r)
    {
      SetProposalError(ModelRotations < 1
        ? "Model range: at least one rotation."
        : "Model range: the comet has no rotation period (Comet tab).");
      return;
    }
    const string Fmt = "yyyy-MM-ddTHH:mm:ss";
    var ci = System.Globalization.CultureInfo.InvariantCulture;
    StartEpoch = r.Start.UtcDateTime.ToString(Fmt, ci);
    EndEpoch = r.End.UtcDateTime.ToString(Fmt, ci);
  }

  /// <summary>The shortest range the timeline accepts, days.</summary>
  public const double MinRangeDays = 1;

  private bool _isProposing;
  /// <summary>
  /// Proposes the boxes' range as soon as it is valid. Invalid or half-typed text is left as it
  /// is, with the reason in <see cref="ErrorMessage"/>, and the last valid proposal stays: the
  /// boxes run this on every keystroke, and snapping them back to the old proposal made a date
  /// impossible to type (feedback #19, "it picks an end date four years away").
  /// </summary>
  private void TryPropose()
  {
    if (_isProposing) return;
    _isProposing = true;
    try
    {
      HasError = false;
      ErrorMessage = string.Empty;
      if (CurrentSession == null) return;

      bool startOk = TimeUtils.TryParseIso8601(StartEpoch, out var startDt);
      bool endOk = TimeUtils.TryParseIso8601(EndEpoch, out var endDt);
      if (!startOk || !endOk)
      {
        SetProposalError(string.Concat(
          !startOk ? "Start" : "End",
          " epoch: use yyyy-MM-dd or yyyy-MM-ddTHH:mm:ss (UTC unless an offset is given)."));
        return;
      }

      var diff = endDt - startDt;
      if (diff.TotalDays < MinRangeDays)
      {
        SetProposalError(diff.TotalDays <= 0
          ? "The end epoch must come after the start epoch."
          : string.Format(System.Globalization.CultureInfo.InvariantCulture,
              "The range must span at least {0} days (now {1:0.##}).", MinRangeDays, diff.TotalDays));
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

  private void SetProposalError(string message)
  {
    ErrorMessage = message;
    HasError = true;
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

    // the range the comet was committed with, not the boxes (a save opens with a later proposal
    // in them); texts that name the same instant are kept as typed
    string start = StartEpoch, end = EndEpoch;
    if (message.Range is { } r)
    {
      start = TimeUtils.EpochText(r.StartCenturies, r.StartNs, StartEpoch);
      end = TimeUtils.EpochText(r.EndCenturies, r.EndNs, EndEpoch);
    }
    _timelineSessionService.UpdateSession(SessionId, s =>
    {
      s.CommittedStartEpoch = start;
      s.CommittedEndEpoch   = end;
    });

    RefreshCommittedState();
    // the boxes show what runs
    StartEpoch = start;
    EndEpoch = end;
    CheckProposedVsCommitted();
    CurrentEpochError = string.Empty;
    PollSimulationClock();
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
