using System;
using System.Reactive.Disposables;
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
/// </summary>
public partial class RangeEntryViewModel : ObservableObject, IDisposable
{
  private readonly ITabStateService<TimelineSession>? _sessions;
  private readonly TimelineService? _timeline;
  private readonly IMessenger? _messenger;
  private readonly Func<double> _rotRateDegDay;
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
  public RangeEntryViewModel(
    ITabStateService<TimelineSession>? sessions,
    TimelineService? timeline,
    IMessenger? messenger,
    Func<double> rotRateDegDay)
  {
    _sessions = sessions;
    _timeline = timeline;
    _messenger = messenger;
    _rotRateDegDay = rotRateDegDay;

    if (_timeline is not null)
    {
      _timeline.ModelRangeRequests
        .Subscribe(end =>
        {
          ModelEndEpoch = end.UtcDateTime.ToString("yyyy-MM-ddTHH:mm:ss", System.Globalization.CultureInfo.InvariantCulture);
          ApplyModelRange();
        })
        .AddDisposableTo(_disposables);
    }

    if (_messenger is not null)
    {
      _messenger.Register<RangeEntryViewModel, CometCommittedMessage>(this, static (r, m) => r.OnCommitted(m.Range));
      _messenger.Register<RangeEntryViewModel, CometDecommittedMessage>(this, static (r, _) => r.OnDecommitted());
    }

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

  // ── Model range (feedback #20): N rotations ending at the image epoch ─────────

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
  /// end box), the period from the comet's rotation rate. Commit as usual.
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
    double rate = _rotRateDegDay();
    double periodHours = rate == 0 || double.IsNaN(rate) ? 0 : 360.0 * 24.0 / Math.Abs(rate);
    if (ModelRange(end, ModelRotations, periodHours) is not { } r)
    {
      SetProposalError(ModelRotations < 1
        ? "Model range: at least one rotation."
        : "Model range: the comet has no rotation period (Rotational Model).");
      return;
    }
    const string Fmt = "yyyy-MM-ddTHH:mm:ss";
    var ci = System.Globalization.CultureInfo.InvariantCulture;
    StartEpoch = r.Start.UtcDateTime.ToString(Fmt, ci);
    EndEpoch = r.End.UtcDateTime.ToString(Fmt, ci);
  }

  // ── Dust lead time (R3) ────────────────────────────────────────────────────

  /// <summary>
  /// Days the jets ignite before the committed start (0 to <see cref="TimelineService.MaxDustLeadDays"/>):
  /// the tail is already formed when play starts. Used by the next commit.
  /// </summary>
  [ObservableProperty]
  private double _dustLeadDays = TimelineService.MaxDustLeadDays;

  private bool _reloading;

  partial void OnDustLeadDaysChanged(double value)
  {
    double applied = Math.Min(TimelineService.MaxDustLeadDays, Math.Max(0, double.IsNaN(value) ? TimelineService.MaxDustLeadDays : value));
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
  /// 2026-10-29), else the proposal; the proposal then follows them.
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

    _reloading = true;
    try { DustLeadDays = session.DustLeadDays ?? TimelineService.MaxDustLeadDays; }
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

  /// <summary>
  /// A commit with <paramref name="range"/> (null: the boxes): the session's committed range is
  /// the range committed, and the boxes show it. Idempotent.
  /// </summary>
  public void OnCommitted(TimeRange? range)
  {
    string start = StartEpoch, end = EndEpoch;
    string typedStart = StartEpoch, typedEnd = EndEpoch;
    Update(s => (start, end) = ApplyCommitted(s, range, typedStart, typedEnd));
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
    Update(s => back = ApplyDecommitted(s));
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
    RefreshProposedState();
    CheckProposedVsCommitted();
  }

  /// <summary>
  /// Writes a commit into <paramref name="s"/>: the committed (and proposed) range is
  /// <paramref name="range"/>, the texts kept as typed when they name the same instant; without a
  /// range, the given texts. Returns the texts.
  /// </summary>
  internal static (string Start, string End) ApplyCommitted(TimelineSession s, TimeRange? range, string startText, string endText)
  {
    string start = startText, end = endText;
    if (range is { } r)
    {
      start = TimeUtils.EpochText(r.StartCenturies, r.StartNs, startText);
      end = TimeUtils.EpochText(r.EndCenturies, r.EndNs, endText);
    }
    s.CommittedStartEpoch = start;
    s.CommittedEndEpoch = end;
    if (TimeUtils.TryParseIso8601(start, out _) && TimeUtils.TryParseIso8601(end, out _))
    {
      s.ProposedStartEpoch = start;
      s.ProposedEndEpoch = end;
    }
    return (start, end);
  }

  /// <summary>
  /// Writes a decommit into <paramref name="s"/>: a committed range becomes the proposal (returned,
  /// for re-proposing), and the committed range is cleared. Null when nothing was committed.
  /// </summary>
  internal static TimeRange? ApplyDecommitted(TimelineSession s)
  {
    TimeRange? back = null;
    if (!string.IsNullOrEmpty(s.CommittedStartEpoch))
    {
      s.ProposedStartEpoch = s.CommittedStartEpoch;
      s.ProposedEndEpoch = s.CommittedEndEpoch;
      back = TryParseRange(s.CommittedStartEpoch, s.CommittedEndEpoch);
    }
    s.CommittedStartEpoch = string.Empty;
    s.CommittedEndEpoch = string.Empty;
    return back;
  }

  private static TimeRange? TryParseRange(string start, string end)
  {
    if (!TimeUtils.TryParseIso8601(start, out var a) || !TimeUtils.TryParseIso8601(end, out var b))
      return null;
    var x = TimeUtils.ToTaiParts(a);
    var y = TimeUtils.ToTaiParts(b);
    return new TimeRange(x.centuries, x.nanoseconds, y.centuries, y.nanoseconds);
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
      var endTai = TimeUtils.ToTaiParts(endDt);
      string start = StartEpoch, end = EndEpoch;
      Update(s =>
      {
        s.ProposedStartEpoch = start;
        s.ProposedEndEpoch = end;
      });
      _timeline?.ProposeEpochRange(new TimeRange(startTai.centuries, startTai.nanoseconds, endTai.centuries, endTai.nanoseconds));
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
    if (_restoredHooked is not null) _restoredHooked.Restored -= Reload;
    _restoredHooked = null;
    _messenger?.UnregisterAll(this);
    _disposables.Dispose();
  }
}
