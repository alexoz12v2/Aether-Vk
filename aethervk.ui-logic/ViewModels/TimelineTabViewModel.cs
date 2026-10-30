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
  private readonly TimeScaleService _timeScale;
  private readonly CompositeDisposable _disposables = [];

  private string _startEpoch = string.Empty;
  /// <summary>
  /// Start of the simulated range, read-only (reviewer feedback item 7: the range is entered in
  /// the Comet tab; the Timeline only shows it): the committed start, else the proposed one.
  /// </summary>
  public string StartEpoch
  {
    get => _startEpoch;
    private set => SetProperty(ref _startEpoch, value);
  }

  private string _endEpoch = string.Empty;
  /// <summary>End of the simulated range, read-only (see <see cref="StartEpoch"/>).</summary>
  public string EndEpoch
  {
    get => _endEpoch;
    private set => SetProperty(ref _endEpoch, value);
  }

  /// <summary>
  /// The Comet tab holds an edited range that differs from the committed one (it applies only
  /// after a Recommit there).
  /// </summary>
  [ObservableProperty]
  private bool _hasPendingRangeEdit;

  public const string PendingRangeEditText =
    "The Comet tab holds a different range: Recommit there to apply it.";

  /// <summary>The shortest range accepted, days (<see cref="RangeEntryViewModel.MinRangeDays"/>).</summary>
  public const double MinRangeDays = RangeEntryViewModel.MinRangeDays;

  /// <inheritdoc cref="RangeEntryViewModel.ModelRange"/>
  internal static (DateTimeOffset Start, DateTimeOffset End)? ModelRange(DateTimeOffset end, int rotations, double periodHours) =>
    RangeEntryViewModel.ModelRange(end, rotations, periodHours);

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
  private bool _hasCommittedState;

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
    }
  }

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

  // ── Time scale (UTC / TDB): the boxes and the overlay label are in the chosen scale ─────

  /// <summary>The scale service (null at construction: TDB, no conversion).</summary>
  public TimeScaleService TimeScale => _timeScale;

  /// <summary>"UTC" or "TDB": the labels ("Start Epoch (UTC):", "Epoch (UTC)").</summary>
  public string ScaleName => _timeScale.ScaleName;

  /// <summary>The labels' tooltip: the scale and its offset to the other one.</summary>
  public string ScaleTip => _timeScale.ScaleTip;

  /// <summary>
  /// The epoch box's UTC date is outside the leap-second table (after its validity, or before
  /// 1972): the conversion may be off by a leap second. A warning glyph next to the box.
  /// </summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(HasCurrentEpochTimeScaleWarning))]
  private string? _currentEpochTimeScaleWarningText;

  public bool HasCurrentEpochTimeScaleWarning => CurrentEpochTimeScaleWarningText is not null;

  /// <summary>The warning's action: dates in TDB (exact), Settings › Time scale for dates.</summary>
  [RelayCommand]
  private void UseTdb() => _timeScale.Scale = Services.TimeScale.Tdb;

  private void OnScaleChanged(TimeScale scale)
  {
    OnPropertyChanged(nameof(ScaleName));
    OnPropertyChanged(nameof(ScaleTip));
    // the session's texts to the new scale (the Comet tab does the same; idempotent), then the display
    if (CurrentSession is not null)
      _timelineSessionService.UpdateSession(SessionId, s => _timeScale.ApplyTo(s));
    RefreshRangeDisplay();
    CurrentEpochError = string.Empty;
    PollSimulationClock(force: true);
  }

  public TimelineTabViewModel(
    ITranslationService translationService,
    ISchedulerProvider schedulerProvider,
    ITabStateService<TimelineSession> sessionService,
    ITabStateService<CometSession> cometSessionService,
    TimelineService timelineService,
    INativeRuntimeService runtimeService,
    ICometMessenger cometMessenger,
    TimeScaleService? timeScale = null)
    : base("Timeline", sessionService, cometMessenger)
  {
    _translationService = translationService;
    _timelineSessionService = sessionService;
    _cometSessionService = cometSessionService;
    _timelineService = timelineService;
    _runtimeService = runtimeService;
    _timeScale = timeScale ?? TimeScaleService.TdbOnly();
    _timeScale.ScaleChanged += OnScaleChanged;
    Icon = "⏱"; // stopwatch — U+23F1

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

    // the range is edited in the Comet tab (the same session): keep the read-only display current
    sessionService.ObserveSession(SessionId)
      .Subscribe(_ => RefreshRangeDisplay())
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
    if (CurrentSession is null || RunToEndTarget(_confirmedRange, CurrentSession.CommittedEndEpoch, _timeScale, CurrentSession) is not var (cent, ns))
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
  /// The epoch box: parses <see cref="CurrentEpochText"/> (in the chosen scale, as displayed),
  /// converts it to the engine's TDB, refuses it outside the committed range, else pauses and
  /// seeks there.
  /// </summary>
  private async Task SeekToEpochAsync()
  {
    if (CurrentSession is null || !HasCommittedState)
      return;
    if (!TryParseEpochBox(CurrentEpochText, _timeScale, out var target, out var status))
    {
      CurrentEpochError = $"Not an epoch: use {EpochBoxFormat} ({ScaleName}).";
      return;
    }
    CurrentEpochTimeScaleWarningText = TimeScaleService.Warns(status) ? _timeScale.WarningText(status) : null;
    if (EffectiveRange() is not { } range)
      return;
    if (!InRange(target, range))
    {
      CurrentEpochError = $"Outside the timeline ({FormatEpochText(range.StartCenturies, range.StartNs)} to {FormatEpochText(range.EndCenturies, range.EndNs)}).";
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

  /// <summary>The epoch box's text read as TDB (no scale service): invariant culture, no offset unless typed.</summary>
  internal static bool TryParseEpochBox(string text, out (short Centuries, ulong Nanoseconds) epoch) =>
    TryParseEpochBox(text, TimeScaleService.TdbOnly(), out epoch, out _);

  /// <summary>
  /// The epoch box's text in the chosen scale as an engine epoch (TDB parts): invariant culture,
  /// the scale's own instant unless an offset is typed. <paramref name="status"/>: the
  /// leap-second status of a UTC text (the date is converted all the same).
  /// </summary>
  internal static bool TryParseEpochBox(string text, TimeScaleService timeScale, out (short Centuries, ulong Nanoseconds) epoch, out UtcStatus status)
  {
    epoch = default;
    status = UtcStatus.Ok;
    if (string.IsNullOrWhiteSpace(text)
        || !DateTimeOffset.TryParse(text.Trim(), System.Globalization.CultureInfo.InvariantCulture,
             System.Globalization.DateTimeStyles.AssumeUniversal, out var dt))
      return false;
    if (!timeScale.IsUtc)
    {
      epoch = TimeUtils.ToTdbParts(dt);
      return true;
    }
    // the text in ISO form: the service parses with the converter
    string iso = dt.ToString("yyyy-MM-dd'T'HH:mm:ss.FFFFFFFK", System.Globalization.CultureInfo.InvariantCulture);
    if (!timeScale.TryParse(iso, out short c, out ulong ns, out status))
      return false;
    epoch = (c, ns);
    return true;
  }

  /// <summary>The engine epoch in <see cref="EpochBoxFormat"/> read as TDB (no scale service).</summary>
  internal static string FormatEpoch(short cent, ulong ns) =>
    TimeUtils.FromTdbParts(cent, ns).ToString(EpochBoxFormat, System.Globalization.CultureInfo.InvariantCulture);

  /// <summary>The engine epoch in <see cref="EpochBoxFormat"/> in the chosen scale.</summary>
  private string FormatEpochText(short cent, ulong ns) => _timeScale.Format(cent, ns, EpochBoxFormat);

  internal static bool InRange((short Centuries, ulong Nanoseconds) e, TimeRange range) =>
    Compare(e, (range.StartCenturies, range.StartNs)) >= 0 && Compare(e, (range.EndCenturies, range.EndNs)) <= 0;

  private static int Compare((short C, ulong N) a, (short C, ulong N) b) =>
    a.C != b.C ? a.C.CompareTo(b.C) : a.N.CompareTo(b.N);

  /// <summary>The engine-confirmed range, else the committed text (in the session's scale).</summary>
  private TimeRange? EffectiveRange()
  {
    if (_confirmedRange is not null)
      return _confirmedRange;
    if (CurrentSession is null)
      return null;
    return RangeEntryViewModel.TryParseRange(CurrentSession.CommittedStartEpoch, CurrentSession.CommittedEndEpoch, _timeScale, CurrentSession);
  }

  /// <summary>
  /// Where Run to End seeks: exactly the engine-confirmed end (the user's end date as the engine
  /// holds it), else the committed end text read as TDB; <c>null</c> without either.
  /// </summary>
  internal static (short Centuries, ulong Nanoseconds)? RunToEndTarget(TimeRange? confirmed, string committedEndEpoch) =>
    RunToEndTarget(confirmed, committedEndEpoch, TimeScaleService.TdbOnly(), null);

  /// <summary>
  /// Where Run to End seeks: exactly the engine-confirmed end, else the committed end text in
  /// <paramref name="session"/>'s scale; <c>null</c> without either.
  /// </summary>
  internal static (short Centuries, ulong Nanoseconds)? RunToEndTarget(TimeRange? confirmed, string committedEndEpoch,
    TimeScaleService timeScale, TimelineSession? session)
  {
    if (confirmed is not null)
      return (confirmed.EndCenturies, confirmed.EndNs);
    var scale = session is null ? timeScale.Scale : timeScale.ScaleOf(session);
    if (timeScale.TryParse(committedEndEpoch, scale, out short c, out ulong ns, out _))
      return (c, ns);
    return null;
  }

  /// <summary>
  /// The viewport's date label for the engine clock (<paramref name="cent"/>, <paramref name="ns"/>):
  /// "yyyy-MM-dd HH:mm (TDB)" with the text read as TDB (no scale service), or empty at the
  /// committed start (and without a committed start).
  /// </summary>
  internal static string ClockLabel(short cent, ulong ns, string committedStartEpoch) =>
    ClockLabel(cent, ns, committedStartEpoch, TimeScaleService.TdbOnly(), Services.TimeScale.Tdb);

  /// <summary>
  /// The viewport's date label for the engine clock: "yyyy-MM-dd HH:mm (UTC)" in the chosen
  /// scale (the suffix names it), or empty at the committed start (and without a committed
  /// start, whose text is in <paramref name="sessionScale"/>).
  /// </summary>
  internal static string ClockLabel(short cent, ulong ns, string committedStartEpoch, TimeScaleService timeScale, TimeScale sessionScale)
  {
    if (!timeScale.TryParse(committedStartEpoch, sessionScale, out short sc, out ulong sn, out _))
      return string.Empty;
    const string Fmt = "yyyy-MM-dd HH:mm";
    var currentStr = timeScale.Format(cent, ns, Fmt);
    return currentStr != timeScale.Format(sc, sn, Fmt) ? currentStr + " (" + timeScale.ScaleName + ")" : string.Empty;
  }

  private void PollSimulationClock() => PollSimulationClock(force: false);

  /// <param name="force">also refresh the epoch box while it is being edited (after its own seek)</param>
  private void PollSimulationClock(bool force)
  {
    if (_runtimeService.GetSimulationClock(out var cent, out var ns))
    {
      if (force || !IsEditingCurrentEpoch)
      {
        CurrentEpochText = FormatEpochText(cent, ns);
        var status = _timeScale.StatusOf(cent, ns);
        CurrentEpochTimeScaleWarningText = TimeScaleService.Warns(status) ? _timeScale.WarningText(status) : null;
      }
      string startEpochStr = CurrentSession?.CommittedStartEpoch ?? "";
      string endEpochStr = CurrentSession?.CommittedEndEpoch ?? "";

      if (string.IsNullOrEmpty(startEpochStr) || string.IsNullOrEmpty(endEpochStr))
      {
         return;
      }

      var sessionScale = CurrentSession is { } cs ? _timeScale.ScaleOf(cs) : _timeScale.Scale;
      if (_timeScale.TryParse(startEpochStr, sessionScale, out short startC, out ulong startN, out _)
          && _timeScale.TryParse(endEpochStr, sessionScale, out short endC, out ulong endN, out _))
      {
         double startTicks = (double)startC * TimeSpan.TicksPerDay * 36525.0 * 1e7 + startN / 100.0;
         double endTicks = (double)endC * TimeSpan.TicksPerDay * 36525.0 * 1e7 + endN / 100.0;
         double currentTicks = (double)cent * TimeSpan.TicksPerDay * 36525.0 * 1e7 + ns / 100.0;

         double progress = (currentTicks - startTicks) / (endTicks - startTicks) * 100.0;
         Progress = Math.Max(0, Math.Min(100, progress));

         // Publish the current epoch string to the session for Overlay window
         var label = ClockLabel(cent, ns, startEpochStr, _timeScale, sessionScale);
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

    // discard the committed range: it flows back into the proposal (the Comet tab's boxes run the
    // same helper on the same message, so the order does not matter)
    TimeRange? back = null;
    _timelineSessionService.UpdateSession(SessionId, s => back = RangeEntryViewModel.ApplyDecommitted(s, _timeScale));
    if (back is not null)
      _timelineService.ProposeEpochRange(back);
    RefreshRangeDisplay();
  }

  /// <summary>The read-only range: the committed one, else the proposal; and whether an edit waits.</summary>
  private void RefreshRangeDisplay()
  {
    var s = CurrentSession;
    if (s is null) return;
    HasCommittedState = !string.IsNullOrEmpty(s.CommittedStartEpoch) && !string.IsNullOrEmpty(s.CommittedEndEpoch);
    if (HasCommittedState)
      DisplayCommittedRange = $"{s.CommittedStartEpoch} to {s.CommittedEndEpoch}";
    StartEpoch = HasCommittedState ? s.CommittedStartEpoch : s.ProposedStartEpoch;
    EndEpoch = HasCommittedState ? s.CommittedEndEpoch : s.ProposedEndEpoch;
    HasPendingRangeEdit = HasCommittedState
      && (s.ProposedStartEpoch != s.CommittedStartEpoch || s.ProposedEndEpoch != s.CommittedEndEpoch);
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

  private void OnSessionRestored() => Restore();

  public override void Dispose()
  {
    _timeScale.ScaleChanged -= OnScaleChanged;
    HookRestored(null);
    _disposables.Dispose();
    base.Dispose();
  }

  /// <summary>Re-reads the session (construction, a savefile open): the range, the speed, the clock.</summary>
  private void Restore()
  {
    if (CurrentSession == null || _timelineSessionService == null) return;

    // the texts in the chosen scale (a save may hold them in the other one); the parts never change
    _timelineSessionService.UpdateSession(SessionId, s => _timeScale.ApplyTo(s));
    RefreshRangeDisplay();

    // the saved speed (a savefile open); none saved keeps the current selection
    if (CurrentSession.Speed is int speed && Enum.IsDefined(typeof(SimulationSpeed), speed))
      SelectedSpeed = (SimulationSpeed)speed;

    // the engine's clock (a reopened save sits at its saved epoch, not at the start)
    if (!IsPlaying) PollSimulationClock();
  }

  private TimeRange? _confirmedRange;

  public void Receive(CometCommittedMessage message)
  {
    if (CurrentSession == null) return;

    // the range the comet was committed with, not the proposal (a save opens with a later one);
    // texts that name the same instant are kept as typed. The Comet tab runs the same helper.
    _timelineSessionService.UpdateSession(SessionId, s =>
      RangeEntryViewModel.ApplyCommitted(s, message.Range, s.ProposedStartEpoch, s.ProposedEndEpoch, _timeScale));

    RefreshRangeDisplay();
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
