using System;
using System.Reactive.Disposables;
using System.Threading.Tasks;
using AetherVk.Logic.Messages;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// The simulated range as the user enters it: the Start/End boxes, the Model range (N rotations
/// ending at an image epoch), the dust lead time, and the lock after a commit with its Recommit
/// and Restore. Lives in the Comet tab (reviewer feedback, item 7: "enter the dates here and make
/// the timeline only show progress"); the Timeline tab shows the committed range read-only.
///
/// <para>Works on the exclusive <see cref="TimelineSession"/> (so saves are unchanged): the boxes
/// are written to its proposed range on every valid keystroke and proposed to
/// <see cref="TimelineService"/>; a commit copies the range it was made with into the committed
/// range (<see cref="ApplyCommitted"/>), a decommit flows it back (<see cref="ApplyDecommitted"/>).
/// The Timeline tab runs the same two helpers on the same messages, so the order does not matter.</para>
///
/// <para>The boxes are typed and shown in the chosen <see cref="TimeScale"/> (UTC by default,
/// <see cref="TimeScaleService"/>); the proposal and the committed range are TDB parts. A scale
/// switch rewrites the texts to the other scale without changing the committed parts.</para>
/// </summary>
public partial class RangeEntryViewModel : ObservableObject, IDisposable
{
  private readonly ITabStateService<TimelineSession>? _sessions;
  private readonly TimelineService? _timeline;
  private readonly IMessenger? _messenger;
  private readonly Func<double> _rotRateDegDay;
  private readonly TimeScaleService _timeScale;
  private readonly CompositeDisposable _disposables = [];
  // used without a session service (design time, tests of the hosting tab alone)
  private readonly TimelineSession _standalone = new();
  private TimelineSession? _restoredHooked;

  /// <summary>Design-time instance: a standalone session, no services.</summary>
  public RangeEntryViewModel() : this(null, null, null, static () => 0) { }

  /// <param name="sessions">the timeline session store (null: a private session, nothing proposed on load)</param>
  /// <param name="timeline">receives the proposals, the lead time and the recommit request</param>
  /// <param name="messenger">comet commit / decommit messages (saves commit without the Comet tab)</param>
  /// <param name="rotRateDegDay">the nucleus rotation rate (°/day) for the Model range</param>
  /// <param name="timeScale">the scale the boxes are in (null: TDB, no conversion)</param>
  public RangeEntryViewModel(
    ITabStateService<TimelineSession>? sessions,
    TimelineService? timeline,
    IMessenger? messenger,
    Func<double> rotRateDegDay,
    TimeScaleService? timeScale = null)
  {
    _sessions = sessions;
    _timeline = timeline;
    _messenger = messenger;
    _rotRateDegDay = rotRateDegDay;
    _timeScale = timeScale ?? TimeScaleService.TdbOnly();

    if (_timeline is not null)
    {
      _timeline.ModelRangeRequests
        .Subscribe(end =>
        {
          // the request is a UTC instant (the image's mid-exposure): the Observation date box is UTC
          ObservationEpoch = FormatUtc(end);
          ApplyModelRange();
        })
        .AddDisposableTo(_disposables);
      // the mode bar's Model: the same chain as the Run at this date button
      _timeline.ObservationRunRequests
        .Subscribe(async request => request.Complete(await RunAtAsync(request.Utc)))
        .AddDisposableTo(_disposables);
    }

    if (_messenger is not null)
    {
      _messenger.Register<RangeEntryViewModel, CometCommittedMessage>(this, static (r, m) => r.OnCommitted(m.Range));
      _messenger.Register<RangeEntryViewModel, CometDecommittedMessage>(this, static (r, _) => r.OnDecommitted());
    }

    _timeScale.ScaleChanged += OnScaleChanged;
    Reload();
  }

  // ── Session ────────────────────────────────────────────────────────────────

  /// <summary>The exclusive timeline session (hooked for <see cref="TimelineSession.Restored"/>).</summary>
  private TimelineSession Session
  {
    get
    {
      var s = _sessions is { ActiveSessionIds.Count: > 0 } svc ? svc.GetSession(svc.ActiveSessionIds[0]) : _standalone;
      if (!ReferenceEquals(_restoredHooked, s))
      {
        if (_restoredHooked is not null) _restoredHooked.Restored -= Reload;
        s.Restored += Reload;
        _restoredHooked = s;
      }
      return s;
    }
  }

  private void Update(Action<TimelineSession> mutate)
  {
    if (_sessions is { ActiveSessionIds.Count: > 0 } svc)
      svc.UpdateSession(svc.ActiveSessionIds[0], mutate);
    else
      mutate(_standalone);
  }

  // ── The boxes ──────────────────────────────────────────────────────────────

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

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(HasValidRange))]
  private bool _hasError;

  [ObservableProperty]
  private string _errorMessage = string.Empty;

  /// <summary>The boxes hold a valid range (it is the proposal the next commit uses).</summary>
  public bool HasValidRange => HasProposedState && !HasError;

  /// <summary>The boxes differ from the committed range.</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(CanRestore), nameof(IsRangeLocked))]
  private bool _isProposedDifferentFromCommitted;

  /// <summary>A committed range exists (the comet is committed).</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(CanRestore), nameof(IsRangeLocked))]
  private bool _hasCommittedState;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(HasValidRange))]
  private bool _hasProposedState;

  [ObservableProperty]
  private string _displayProposedRange = string.Empty;

  [ObservableProperty]
  private string _displayCommittedRange = string.Empty;

  public bool CanRestore => HasCommittedState && IsProposedDifferentFromCommitted;

  /// <summary>
  /// The boxes were edited after a commit: the engine keeps the committed range (reviewer report
  /// 2026-10-29, "it picks other dates") until the comet is committed again with the new one.
  /// </summary>
  public bool IsRangeLocked => HasCommittedState && IsProposedDifferentFromCommitted;

  public const string RangeLockedText =
    "Range locked while the comet is committed: the simulation keeps the committed range. Decommit to change it, or recommit with this range.";

  /// <summary>The shortest range accepted, days.</summary>
  public const double MinRangeDays = 1;

  /// <summary>Commits the comet again with the boxes' range (the Comet tab's Download &amp; Commit).</summary>
  [RelayCommand]
  private void RecommitWithRange()
  {
    if (IsRangeLocked && !HasError)
      _timeline?.RequestRecommit();
  }

  /// <summary>The Restore button: the boxes back to the committed range (else the saved proposal).</summary>
  [RelayCommand]
  private void Restore() => Reload();

  // ── Time scale (UTC / TDB) ─────────────────────────────────────────────────

  /// <summary>The scale service (the Comet tab's other date texts use it too).</summary>
  public TimeScaleService TimeScale => _timeScale;

  /// <summary>"UTC" or "TDB": the boxes' labels ("Start (UTC):").</summary>
  public string ScaleName => _timeScale.ScaleName;

  /// <summary>The labels' tooltip: the scale and its offset to the other one.</summary>
  public string ScaleTip => _timeScale.ScaleTip;

  /// <summary>
  /// A box's UTC date is outside the leap-second table (after its validity, or before 1972): the
  /// conversion may be off by a leap second. Shown as a warning glyph next to the boxes.
  /// </summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(HasTimeScaleWarning))]
  private string? _timeScaleWarningText;

  public bool HasTimeScaleWarning => TimeScaleWarningText is not null;

  /// <summary>The leap-second status of the boxes (the worse of the two).</summary>
  public UtcStatus TimeScaleStatus { get; private set; } = UtcStatus.Ok;

  /// <summary>The warning's action: dates in TDB (exact), Settings › Time scale for dates.</summary>
  [RelayCommand]
  private void UseTdb() => _timeScale.Scale = Services.TimeScale.Tdb;

  private void OnScaleChanged(TimeScale scale)
  {
    OnPropertyChanged(nameof(ScaleName));
    OnPropertyChanged(nameof(ScaleTip));
    // the texts to the new scale, the committed parts unchanged; then the texts back into the boxes
    Update(s => _timeScale.ApplyTo(s));
    ReloadBoxes();
  }

  private static UtcStatus Worse(UtcStatus a, UtcStatus b) => (int)a >= (int)b ? a : b;

  // ── Observation (feedback #20, reviewer slide 1 of 2026-10-31): the date and N rotations before it ──

  /// <summary>The default N rotations (old saves, a fresh session).</summary>
  public const int DefaultRotations = 3;

  /// <summary>The box format of the observation date, UTC.</summary>
  public const string UtcBoxFormat = "yyyy-MM-ddTHH:mm:ss";

  /// <summary>
  /// The observation date, UTC ("yyyy-MM-ddTHH:mm:ss", the image's mid-exposure); empty: the end
  /// box's instant. Separate from the generation period (N rotations) so another date over the
  /// weeks of a campaign does not mean redoing the model.
  /// </summary>
  [ObservableProperty]
  private string _observationEpoch = string.Empty;

  partial void OnObservationEpochChanged(string value)
  {
    if (!_reloading)
      Update(s => s.ObservationEpochUtc = value);
  }

  /// <summary>Nucleus rotations the range spans before the observation date (N ≥ 1).</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(ModelSpanText))]
  private int _modelRotations = DefaultRotations;

  partial void OnModelRotationsChanged(int value)
  {
    if (!_reloading)
      Update(s => s.ObservationRotations = value);
  }

  /// <summary>The rotation period from the comet's rate, hours (0: none).</summary>
  private double PeriodHours()
  {
    double rate = _rotRateDegDay();
    return rate == 0 || double.IsNaN(rate) || double.IsInfinity(rate) ? 0 : 360.0 * 24.0 / Math.Abs(rate);
  }

  /// <summary>"= x d": the span N rotations make (at least <see cref="MinRangeDays"/>); empty without a period.</summary>
  public string ModelSpanText
  {
    get
    {
      double periodHours = PeriodHours();
      if (ModelRange(new DateTimeOffset(2000, 1, 1, 12, 0, 0, TimeSpan.Zero), ModelRotations, periodHours) is not { } r)
        return string.Empty;
      double days = (r.End - r.Start).TotalDays;
      bool floored = ModelRotations * periodHours < MinRangeDays * 24.0;
      return string.Format(System.Globalization.CultureInfo.InvariantCulture, floored ? "= {0:0.##} d (minimum)" : "= {0:0.##} d", days);
    }
  }

  /// <summary>The rotation rate changed (the Comet tab's Rotational Model): the span text follows.</summary>
  public void RefreshModelSpan() => OnPropertyChanged(nameof(ModelSpanText));

  /// <summary>A UTC instant in the observation box format.</summary>
  public static string FormatUtc(DateTimeOffset utc) =>
    utc.UtcDateTime.ToString(UtcBoxFormat, System.Globalization.CultureInfo.InvariantCulture);

  /// <summary>The loaded image's date (a FITS billboard with an observation date), or null.</summary>
  public DateTimeOffset? ImageObservationUtc() => _timeline?.ObservationImage()?.ObservationUtc;

  /// <summary>"From image": the observation date of the loaded FITS image into the box.</summary>
  [RelayCommand]
  private void FromImage()
  {
    var image = _timeline?.ObservationImage();
    if (image?.ObservationUtc is { } utc)
    {
      ObservationEpoch = FormatUtc(utc);
      RunStatus = string.Empty;
    }
    else
    {
      RunStatus = image is null ? NoImageText : FitsWithoutDateText(image);
    }
  }

  /// <summary>No telescope image is loaded (AppStrings, the UI culture).</summary>
  public static string NoImageText => GetString(ObservationImageText.NoImageKey);

  /// <summary>A FITS image is loaded without a readable date: the text names its date card, else
  /// the keywords tried (AppStrings, the UI culture).</summary>
  public static string FitsWithoutDateText(IObservationImage image) =>
    ObservationImageText.FormatWithoutDate(GetString(ObservationImageText.FitsWithoutDateKey), image, GetString);

  private static System.Resources.ResourceManager? _strings;

  // the app's TranslationService sets CurrentUICulture on every language change
  private static string GetString(string key)
  {
    _strings ??= new System.Resources.ResourceManager("AetherVk.Logic.Resources.AppStrings", typeof(ITranslationService).Assembly);
    return _strings.GetString(key, System.Globalization.CultureInfo.CurrentUICulture) ?? string.Concat("[", key, "]");
  }

  // ── Run at this date: propose → recommit → seek → compare ──────────────────

  /// <summary>The chain is running (the block shows a spinner; the buttons wait).</summary>
  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(RunAtDateCommand))]
  private bool _isRunning;

  /// <summary>The chain's progress, then its outcome (one line under the block).</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(HasRunStatus))]
  private string _runStatus = string.Empty;

  public bool HasRunStatus => RunStatus.Length > 0;

  private bool CanRun() => !IsRunning;

  /// <summary>
  /// The button: N rotations ending at the observation date are proposed, the comet is committed
  /// again with them (the SPK is downloaded only when the current one does not cover the range),
  /// the clock is sought to the date, and the comparison with the loaded image is set up.
  /// </summary>
  [RelayCommand(CanExecute = nameof(CanRun))]
  private Task RunAtDateAsync() => RunAtAsync(null);

  /// <summary>
  /// The one implementation of the chain (the button, the mode bar's Model through
  /// <see cref="TimelineService.RunAtObservationAsync"/>): <paramref name="utc"/> replaces the box
  /// when given. Never throws; the outcome is in the result and in <see cref="RunStatus"/>.
  /// </summary>
  public async Task<ObservationRunResult> RunAtAsync(DateTimeOffset? utc)
  {
    if (IsRunning)
      return new ObservationRunResult(false, false, false, false, "A run is already in progress.");
    IsRunning = true;
    bool proposed = false, committed = false, seeked = false, compared = false;
    try
    {
      if (utc is { } given)
        ObservationEpoch = FormatUtc(given);
      RunStatus = "Proposing the range…";
      if (!TryApplyModelRange() || HasError)
        return Finish(proposed, committed, seeked, compared, ErrorMessage);
      proposed = true;
      if (_timeline is null)
        return Finish(proposed, committed, seeked, compared, "No timeline service.");
      if (!TryObservationTdb(out short c, out ulong ns, out string? reason))
        return Finish(proposed, committed, seeked, compared, reason ?? "Observation date: not a date.");

      RunStatus = "Committing the comet with this range…";
      committed = await _timeline.RequestRecommitAsync();
      if (!committed)
        return Finish(proposed, committed, seeked, compared,
          "The comet was not committed: pick a comet and an SPK record in this tab, then run again.");

      RunStatus = "Seeking to the observation date…";
      seeked = await _timeline.SeekAsync(c, ns);
      if (!seeked)
        return Finish(proposed, committed, seeked, compared, "Committed, but the seek to the date was refused.");

      var image = _timeline.ObservationImage();
      if (image is null)
        return Finish(proposed, committed, seeked, compared, string.Concat("Committed and at ", ObservationEpoch, " UTC. ", NoImageText));
      if (!image.CanSetUpComparison)
        return Finish(proposed, committed, seeked, compared, string.Concat("Committed and at ", ObservationEpoch, " UTC; the image has no viewport to compare in."));
      RunStatus = "Setting up the comparison with the image…";
      await image.SetUpComparisonAsync();
      compared = true;
      return Finish(proposed, committed, seeked, compared, image.StatusMessage);
    }
    catch (Exception ex)
    {
      return Finish(proposed, committed, seeked, compared, string.Concat("Error: ", ex.Message));
    }
    finally
    {
      IsRunning = false;
    }
  }

  private ObservationRunResult Finish(bool proposed, bool committed, bool seeked, bool compared, string status)
  {
    RunStatus = status;
    return new ObservationRunResult(proposed, committed, seeked, compared, status);
  }

  /// <summary>The observation box (or the end box when empty) as TDB parts; false with the reason.</summary>
  private bool TryObservationTdb(out short centuries, out ulong nanoseconds, out string? reason)
  {
    centuries = 0;
    nanoseconds = 0;
    reason = null;
    if (string.IsNullOrWhiteSpace(ObservationEpoch))
    {
      if (_timeScale.TryParse(EndEpoch, out centuries, out nanoseconds, out _))
        return true;
      reason = "The end epoch is not a date.";
      return false;
    }
    if (!TimeUtils.TryParseIso8601(ObservationEpoch, out var utc))
    {
      reason = "Observation date: use yyyy-MM-ddTHH:mm:ss (UTC).";
      return false;
    }
    try
    {
      (centuries, nanoseconds) = _timeScale.FromUtc(utc);
      return true;
    }
    catch (Exception ex)
    {
      reason = string.Concat("Observation date: ", ex.Message);
      return false;
    }
  }

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
  /// Proposes <see cref="ModelRotations"/> rotations ending at <see cref="ObservationEpoch"/> (UTC;
  /// empty: the end box), the period from the comet's rotation rate. Commit as usual (Run at this
  /// date does). The span is measured in TDB (uniform time); the boxes show both ends in the
  /// chosen scale. False (with <see cref="ErrorMessage"/>) when nothing was proposed.
  /// </summary>
  [RelayCommand]
  private void ApplyModelRange() => TryApplyModelRange();

  /// <summary>The command's body: the boxes from the Observation block; false when nothing was proposed.</summary>
  public bool TryApplyModelRange()
  {
    short ec;
    ulong en;
    if (string.IsNullOrWhiteSpace(ObservationEpoch))
    {
      if (!_timeScale.TryParse(EndEpoch, out ec, out en, out _))
      {
        SetProposalError("Observation date: enter a date (yyyy-MM-ddTHH:mm:ss, UTC), or a valid End epoch.");
        return false;
      }
    }
    else if (!TryObservationTdb(out ec, out en, out string? reason))
    {
      SetProposalError(reason ?? "Observation date: not a date.");
      return false;
    }
    double periodHours = PeriodHours();
    var endTdb = TimeUtils.FromTdbParts(ec, en);
    if (ModelRange(endTdb, ModelRotations, periodHours) is not { } r)
    {
      SetProposalError(ModelRotations < 1
        ? "Model range: at least one rotation."
        : "Model range: the comet has no rotation period (Rotational Model).");
      return false;
    }
    var (sc, sn) = TimeUtils.ToTdbParts(r.Start);
    // both boxes first, then one proposal (a box-by-box proposal would persist a mixed range)
    _suppressPropose = true;
    try
    {
      StartEpoch = _timeScale.Format(sc, sn, TimeScaleService.IsoSecondsFormat);
      EndEpoch = _timeScale.Format(ec, en, TimeScaleService.IsoSecondsFormat);
    }
    finally
    {
      _suppressPropose = false;
    }
    TryPropose();
    return !HasError;
  }

  // ── Dust lead time (R3) ────────────────────────────────────────────────────

  /// <summary>
  /// Days the jets ignite before the committed start (0 to the longest jet lifetime,
  /// <see cref="TimelineService.MaxDustLeadDaysNow"/>): the tail is already formed when play
  /// starts. Used by the next commit.
  /// </summary>
  [ObservableProperty]
  private double _dustLeadDays = TimelineService.MaxDustLeadDays;

  private bool _reloading;

  partial void OnDustLeadDaysChanged(double value)
  {
    double max = _timeline?.MaxDustLeadDaysNow ?? TimelineService.MaxDustLeadDays;
    double applied = Math.Min(max, Math.Max(0, double.IsNaN(value) ? Math.Min(TimelineService.MaxDustLeadDays, max) : value));
    if (_timeline is not null)
    {
      _timeline.DustLeadDays = value;
      applied = _timeline.DustLeadDays;
    }
    if (applied != value)
    {
      DustLeadDays = applied; // clamped
      return;
    }
    if (!_reloading)
      Update(s => s.DustLeadDays = applied);
  }

  // ── Load, commit, decommit ─────────────────────────────────────────────────

  /// <summary>
  /// Re-reads the session (construction, Restore, a savefile open): the boxes show the committed
  /// range when there is one (a save keeps a later, uncommitted proposal: reviewer report
  /// 2026-10-29), else the proposal; the proposal then follows them. The texts are first brought
  /// to the chosen scale (a save may hold them in the other one).
  /// </summary>
  public void Reload()
  {
    var session = Session;
    if (string.IsNullOrEmpty(session.ProposedStartEpoch) || string.IsNullOrEmpty(session.ProposedEndEpoch))
    {
      Update(s =>
      {
        if (string.IsNullOrEmpty(s.ProposedStartEpoch)) s.ProposedStartEpoch = TimelineSession.DefaultProposedStart;
        if (string.IsNullOrEmpty(s.ProposedEndEpoch)) s.ProposedEndEpoch = TimelineSession.DefaultProposedEnd;
      });
    }
    // pins the texts' scale (so a later switch knows what to rewrite from) and rewrites a save's
    // texts typed in the other scale; the committed parts never change
    Update(s => _timeScale.ApplyTo(s));

    ReloadBoxes();

    _reloading = true;
    try
    {
      DustLeadDays = session.DustLeadDays ?? TimelineService.MaxDustLeadDays;
      // the Observation block: a save without it (older) shows N = 3 and the committed end as UTC
      ModelRotations = session.ObservationRotations is { } n && n >= 1 ? n : DefaultRotations;
      ObservationEpoch = session.ObservationEpochUtc ?? CommittedEndUtcText(session);
    }
    finally { _reloading = false; }

    HasError = false;
    ErrorMessage = string.Empty;
    // a private session (no session service) proposes only when edited
    if (_sessions is not null)
      TryPropose();
    RefreshCommittedState();
    RefreshProposedState();
    CheckProposedVsCommitted();
  }

  /// <summary>The session's committed end as a UTC box text; empty without one or without a converter.</summary>
  private string CommittedEndUtcText(TimelineSession session)
  {
    if (string.IsNullOrEmpty(session.CommittedEndEpoch))
      return string.Empty;
    if (!_timeScale.TryParse(session.CommittedEndEpoch, _timeScale.ScaleOf(session), out short c, out ulong ns, out _))
      return string.Empty;
    try
    {
      string text = _timeScale.Format(c, ns, UtcBoxFormat, Services.TimeScale.Utc);
      // without a converter the text is the TDB one labelled " TDB": not a UTC date
      return TimeUtils.TryParseIso8601(text, out _) ? text : string.Empty;
    }
    catch
    {
      return string.Empty;
    }
  }

  /// <summary>The boxes from the session's texts (the committed range, else the proposal), without proposing.</summary>
  private void ReloadBoxes()
  {
    var session = Session;
    RefreshCommittedState();
    // both boxes first, then one proposal: a box-by-box proposal would persist a mixed range
    _suppressPropose = true;
    try
    {
      StartEpoch = HasCommittedState ? session.CommittedStartEpoch : session.ProposedStartEpoch;
      EndEpoch = HasCommittedState ? session.CommittedEndEpoch : session.ProposedEndEpoch;
    }
    finally
    {
      _suppressPropose = false;
    }
    RefreshTimeScaleWarning();
    RefreshProposedState();
    CheckProposedVsCommitted();
  }

  /// <summary>
  /// A commit with <paramref name="range"/> (null: the boxes): the session's committed range is
  /// the range committed, and the boxes show it. Idempotent.
  /// </summary>
  public void OnCommitted(TimeRange? range)
  {
    string start = StartEpoch, end = EndEpoch;
    string typedStart = StartEpoch, typedEnd = EndEpoch;
    Update(s => (start, end) = ApplyCommitted(s, range, typedStart, typedEnd, _timeScale));
    RefreshCommittedState();
    _suppressPropose = true;
    try
    {
      StartEpoch = start;
      EndEpoch = end;
    }
    finally
    {
      _suppressPropose = false;
    }
    TryPropose();
    RefreshProposedState();
    CheckProposedVsCommitted();
  }

  /// <summary>A decommit: the committed range flows back into the proposal and the boxes. Idempotent.</summary>
  public void OnDecommitted()
  {
    TimeRange? back = null;
    Update(s => back = ApplyDecommitted(s, _timeScale));
    if (back is not null)
      _timeline?.ProposeEpochRange(back);
    var session = Session;
    RefreshCommittedState();
    _suppressPropose = true;
    try
    {
      StartEpoch = session.ProposedStartEpoch;
      EndEpoch = session.ProposedEndEpoch;
    }
    finally
    {
      _suppressPropose = false;
    }
    HasError = false;
    ErrorMessage = string.Empty;
    RefreshTimeScaleWarning();
    RefreshProposedState();
    CheckProposedVsCommitted();
  }

  /// <summary>
  /// Writes a commit into <paramref name="s"/>: the committed (and proposed) range is
  /// <paramref name="range"/>, the texts kept as typed when they name the same instant (in the
  /// session's scale); without a range, the given texts. Returns the texts.
  /// </summary>
  internal static (string Start, string End) ApplyCommitted(TimelineSession s, TimeRange? range, string startText, string endText,
    TimeScaleService? timeScale = null)
  {
    var ts = timeScale ?? TimeScaleService.TdbOnly();
    var scale = ts.ScaleOf(s);
    string start = startText, end = endText;
    if (range is { } r)
    {
      start = ts.EpochText(r.StartCenturies, r.StartNs, startText, scale);
      end = ts.EpochText(r.EndCenturies, r.EndNs, endText, scale);
    }
    s.CommittedStartEpoch = start;
    s.CommittedEndEpoch = end;
    if (TimeUtils.TryParseIso8601(start, out _) && TimeUtils.TryParseIso8601(end, out _))
    {
      s.ProposedStartEpoch = start;
      s.ProposedEndEpoch = end;
    }
    s.ProposedScale = TimeScaleService.KeyOf(scale);
    return (start, end);
  }

  /// <summary>
  /// Writes a decommit into <paramref name="s"/>: a committed range becomes the proposal (returned,
  /// for re-proposing), and the committed range is cleared. Null when nothing was committed.
  /// </summary>
  internal static TimeRange? ApplyDecommitted(TimelineSession s, TimeScaleService? timeScale = null)
  {
    TimeRange? back = null;
    if (!string.IsNullOrEmpty(s.CommittedStartEpoch))
    {
      s.ProposedStartEpoch = s.CommittedStartEpoch;
      s.ProposedEndEpoch = s.CommittedEndEpoch;
      back = TryParseRange(s.CommittedStartEpoch, s.CommittedEndEpoch, timeScale ?? TimeScaleService.TdbOnly(), s);
    }
    s.CommittedStartEpoch = string.Empty;
    s.CommittedEndEpoch = string.Empty;
    return back;
  }

  /// <summary>The range of two texts in <paramref name="s"/>'s scale; null when one is no date.</summary>
  internal static TimeRange? TryParseRange(string start, string end, TimeScaleService timeScale, TimelineSession s)
  {
    var scale = timeScale.ScaleOf(s);
    if (!timeScale.TryParse(start, scale, out short sc, out ulong sn, out _)
        || !timeScale.TryParse(end, scale, out short ec, out ulong en, out _))
      return null;
    return new TimeRange(sc, sn, ec, en);
  }

  private void RefreshCommittedState()
  {
    var s = Session;
    HasCommittedState = !string.IsNullOrEmpty(s.CommittedStartEpoch) && !string.IsNullOrEmpty(s.CommittedEndEpoch);
    DisplayCommittedRange = HasCommittedState ? $"{s.CommittedStartEpoch} to {s.CommittedEndEpoch}" : string.Empty;
  }

  private void RefreshProposedState()
  {
    var s = Session;
    HasProposedState = !string.IsNullOrEmpty(s.ProposedStartEpoch) && !string.IsNullOrEmpty(s.ProposedEndEpoch);
    DisplayProposedRange = HasProposedState ? $"{s.ProposedStartEpoch} to {s.ProposedEndEpoch}" : string.Empty;
  }

  private void CheckProposedVsCommitted()
  {
    var s = Session;
    IsProposedDifferentFromCommitted = StartEpoch != s.CommittedStartEpoch || EndEpoch != s.CommittedEndEpoch;
  }

  /// <summary>The warning from the boxes' texts in the chosen scale.</summary>
  private void RefreshTimeScaleWarning()
  {
    var status = Worse(_timeScale.StatusOf(StartEpoch), _timeScale.StatusOf(EndEpoch));
    TimeScaleStatus = status;
    TimeScaleWarningText = TimeScaleService.Warns(status) ? _timeScale.WarningText(status) : null;
  }

  // ── Proposal ───────────────────────────────────────────────────────────────

  private bool _isProposing;
  private bool _suppressPropose;

  /// <summary>
  /// Proposes the boxes' range as soon as it is valid. Invalid or half-typed text is left as it
  /// is, with the reason in <see cref="ErrorMessage"/>, and the last valid proposal stays: the
  /// boxes run this on every keystroke, and snapping them back to the old proposal made a date
  /// impossible to type (feedback #19, "it picks an end date four years away").
  /// </summary>
  private void TryPropose()
  {
    if (_isProposing || _suppressPropose) return;
    _isProposing = true;
    try
    {
      HasError = false;
      ErrorMessage = string.Empty;

      bool startOk = _timeScale.TryParse(StartEpoch, out short sc, out ulong sn, out var startStatus);
      bool endOk = _timeScale.TryParse(EndEpoch, out short ec, out ulong en, out var endStatus);
      var status = Worse(startStatus, endStatus);
      TimeScaleStatus = status;
      TimeScaleWarningText = startOk && endOk && TimeScaleService.Warns(status) ? _timeScale.WarningText(status) : null;
      if (!startOk || !endOk)
      {
        SetProposalError(string.Concat(
          !startOk ? "Start" : "End",
          " epoch: use yyyy-MM-dd or yyyy-MM-ddTHH:mm:ss (", ScaleName, " unless an offset is given)."));
        return;
      }

      var diff = TimeUtils.FromTdbParts(ec, en) - TimeUtils.FromTdbParts(sc, sn);
      if (diff.TotalDays < MinRangeDays)
      {
        SetProposalError(diff.TotalDays <= 0
          ? "The end epoch must come after the start epoch."
          : string.Format(System.Globalization.CultureInfo.InvariantCulture,
              "The range must span at least {0} days (now {1:0.##}).", MinRangeDays, diff.TotalDays));
        return;
      }

      string start = StartEpoch, end = EndEpoch;
      string scaleKey = TimeScaleService.KeyOf(_timeScale.Scale);
      Update(s =>
      {
        s.ProposedStartEpoch = start;
        s.ProposedEndEpoch = end;
        s.ProposedScale = scaleKey;
      });
      _timeline?.ProposeEpochRange(new TimeRange(sc, sn, ec, en));
      RefreshProposedState();
      CheckProposedVsCommitted();
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

  public void Dispose()
  {
    _timeScale.ScaleChanged -= OnScaleChanged;
    if (_restoredHooked is not null) _restoredHooked.Restored -= Reload;
    _restoredHooked = null;
    _messenger?.UnregisterAll(this);
    _disposables.Dispose();
  }
}
