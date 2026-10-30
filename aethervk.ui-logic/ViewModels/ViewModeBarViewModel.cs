using System;
using System.Linq;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Attributes;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// The main window's view-mode strip (reviewer items 4 and 20): three buttons, Orbit / Nucleus /
/// Model, acting on the active viewport's camera through the same <see cref="CameraService"/> calls
/// the viewport overlay's buttons make. Model also runs the FITS comparison when an image billboard
/// is there: the Timeline's model range ending at the image's date (N rotations back), then
/// "Set up comparison"; without one it enters the Earth observer and shows <see cref="Hint"/>.
/// The Timeline's range lock (a committed comet whose range would change) is surfaced as its text
/// and its "Recommit with this range" button; it is never bypassed.
/// </summary>
[GenerateLocalizedStrings(keyPrefix: "ModeBar_", designTitle: "View mode")]
public partial class ViewModeBarViewModel : ObservableObject, IDisposable
{
  private readonly ITranslationService _translationService;
  private readonly ICameraServiceRegistry _cameras;
  private readonly TimelineService _timeline;
  private readonly ITabStateService<TimelineSession>? _timelineSessions;
  private readonly CompositeDisposable _disposables = [];
  private readonly SerialDisposable _modeSubscription = new();
  private CameraService? _activeCamera;
  private DesktopBillboardsViewModel? _billboards;

  /// <summary>How often the Timeline's range lock is re-read (its tab owns the state).</summary>
  public static readonly TimeSpan RangeLockPollInterval = TimeSpan.FromMilliseconds(250);

  /// <summary>The mode the active camera is in (the highlight follows every mode change).</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(IsOrbitActive), nameof(IsNucleusActive), nameof(IsModelActive))]
  private CameraMode _currentMode = CameraMode.UpZenith;

  public bool IsOrbitActive => CurrentMode == CameraMode.UpZenith;
  public bool IsNucleusActive => CurrentMode == CameraMode.CometOrbiting;
  public bool IsModelActive => CurrentMode == CameraMode.EarthPosition;

  /// <summary>One line after Model: the comparison's outcome, or the no-FITS hint.</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(HasHint))]
  private string _hint = string.Empty;

  public bool HasHint => Hint.Length > 0;

  /// <summary>The Timeline's R2 lock: the comet is committed and the proposed range differs.</summary>
  [ObservableProperty]
  private bool _isRangeLocked;

  public string RangeLockText => RangeEntryViewModel.RangeLockedText;

  /// <summary>A Model comparison is running (the buttons wait for it).</summary>
  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(SelectOrbitCommand), nameof(SelectNucleusCommand), nameof(SelectModelCommand))]
  private bool _isBusy;

  public ViewModeBarViewModel(
    ITranslationService translationService,
    ISchedulerProvider schedulerProvider,
    ICameraServiceRegistry cameraServiceRegistry,
    TimelineService timelineService,
    ITabStateService<TimelineSession>? timelineSessions = null,
    bool pollRangeLock = true)
  {
    _translationService = translationService;
    _cameras = cameraServiceRegistry;
    _timeline = timelineService;
    _timelineSessions = timelineSessions;
    _disposables.Add(_modeSubscription);

    RefreshStrings();
    _translationService.CultureChanged.Skip(1)
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(_ => RefreshStrings())
      .AddDisposableTo(_disposables);

    // a viewport opened or closed: follow the camera that is now the active one
    _cameras.ViewportCreated.Merge(_cameras.ViewportDestroyed)
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(_ => BindCamera())
      .AddDisposableTo(_disposables);

    // the Timeline tab owns the lock state: re-read it periodically (tests call RefreshRangeLock)
    if (pollRangeLock)
      Observable.Interval(RangeLockPollInterval, schedulerProvider.MainThread)
        .Subscribe(_ => RefreshRangeLock())
        .AddDisposableTo(_disposables);

    BindCamera();
  }

  /// <summary>The focused viewport's camera (the main window sets it); null falls back to the
  /// first registered camera.</summary>
  public CameraService? ActiveCamera
  {
    get => _activeCamera;
    set
    {
      if (ReferenceEquals(_activeCamera, value))
        return;
      _activeCamera = value;
      BindCamera();
    }
  }

  /// <summary>The camera the buttons act on.</summary>
  public CameraService? Camera => _activeCamera ?? _cameras.All.FirstOrDefault();

  /// <summary>The billboards Model looks through for a FITS image (the Settings tab's rows);
  /// disposing the result detaches them.</summary>
  public IDisposable AttachBillboards(DesktopBillboardsViewModel billboards)
  {
    _billboards = billboards;
    return Disposable.Create(() =>
    {
      if (ReferenceEquals(_billboards, billboards))
        _billboards = null;
    });
  }

  private void BindCamera()
  {
    var camera = Camera;
    if (camera is null)
    {
      _modeSubscription.Disposable = null;
      return;
    }
    _modeSubscription.Disposable = camera.CameraModeChanged.Subscribe(mode => CurrentMode = mode);
    CurrentMode = camera.CurrentMode;
  }

  private bool CanSelect() => !IsBusy;

  /// <summary>Orbit: the ecliptic seen from above (what the overlay's Orbit button does).</summary>
  [RelayCommand(CanExecute = nameof(CanSelect))]
  private void SelectOrbit() => Camera?.SetCameraMode(CameraMode.UpZenith);

  /// <summary>Nucleus: orbit the nucleus in its body frame (the overlay's Nucleus button).</summary>
  [RelayCommand(CanExecute = nameof(CanSelect))]
  private void SelectNucleus() => Camera?.SetCameraMode(CameraMode.CometOrbiting);

  /// <summary>
  /// Model: with a FITS billboard, the Timeline's model range ending at the image's date, then
  /// "Set up comparison" (Earth observer, seek, site, match, centre, lock); without one, the Earth
  /// observer (the overlay's Model button) and the import hint.
  /// </summary>
  [RelayCommand(CanExecute = nameof(CanSelect))]
  private async Task SelectModelAsync()
  {
    if (Camera is not { } camera)
      return;
    var fits = FitsImage();
    if (fits is null)
    {
      camera.EnterModelView();
      Hint = StrNoFitsHint;
      RefreshRangeLock();
      return;
    }
    IsBusy = true;
    try
    {
      Hint = string.Empty;
      // end = the image's mid-exposure date, start = N rotations back (the Timeline's N)
      if (fits.RequestModelRangeCommand.CanExecute(null))
        fits.RequestModelRangeCommand.Execute(null);
      await fits.SetUpComparisonCommand.ExecuteAsync(null);
      Hint = fits.StatusMessage;
    }
    finally
    {
      IsBusy = false;
    }
    RefreshRangeLock();
  }

  /// <summary>The first FITS billboard, one with an observation date first.</summary>
  private FitsImageViewModel? FitsImage() =>
    _billboards?.Items
      .Select(i => i.Fits)
      .Where(f => f is not null)
      .OrderByDescending(f => f!.ObservationUtc is not null)
      .FirstOrDefault();

  /// <summary>The Timeline's "Recommit with this range" (the Comet tab's Download &amp; Commit).</summary>
  [RelayCommand]
  private void RecommitWithRange()
  {
    if (IsRangeLocked)
      _timeline.RequestRecommit();
  }

  /// <summary>Re-reads the Timeline sessions: locked when a committed range differs from the proposed one.</summary>
  public void RefreshRangeLock()
  {
    if (_timelineSessions is not { } sessions)
      return;
    bool locked = false;
    foreach (var id in sessions.ActiveSessionIds)
    {
      var s = sessions.GetSession(id);
      if (s is null || s.CommittedStartEpoch.Length == 0)
        continue;
      if (s.ProposedStartEpoch != s.CommittedStartEpoch || s.ProposedEndEpoch != s.CommittedEndEpoch)
      {
        locked = true;
        break;
      }
    }
    IsRangeLocked = locked;
  }

  public void Dispose() => _disposables.Dispose();
}
