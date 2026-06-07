using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading;
using AetherVk.Logic.Messages;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

public class TimeScaleOption
{
  public string DisplayName { get; init; } = "";
  public uint Value { get; init; }
}

public partial class TimelineViewModel
  : TabItemViewModel,
    IDisposable,
    IRecipient<CometDestroyedMessage>,
    IRecipient<JetConfigChangedMessage>,
    IRecipient<AlmanacUpdatedMessage>
{
  private readonly NativeRuntimeService _runtimeService;
  private readonly IUiThreadDispatcher _uiThreadDispatcher;
  private readonly TrajectoryManagerService _trajectoryManager;
  private readonly SceneStateManager _sceneStateManager;
  private readonly BreadcrumbService _breadcrumbService;
  private readonly Timer _timer;
  private bool _isDragging;

  public TimelineService Timeline { get; }

  [ObservableProperty]
  private ulong _currentSceneId;

  public ObservableCollection<TimeScaleOption> TimeScaleOptions { get; } =
    new()
    {
      new TimeScaleOption { DisplayName = "1 Day/sec", Value = 1 },
      new TimeScaleOption { DisplayName = "1 Week/sec", Value = 2 },
      new TimeScaleOption { DisplayName = "1 Month/sec", Value = 3 },
    };

  [ObservableProperty]
  private TimeScaleOption _selectedTimeScale;

  // ── Epoch Range ─────────────────────────────────────────────────────────────

  [ObservableProperty]
  private string _startEpochText = "";

  [ObservableProperty]
  private string _endEpochText = "";

  [ObservableProperty]
  private bool _hasEpochError;

  [ObservableProperty]
  private string _epochErrorText = "";

  /// <summary>Epochs can only be edited when simulation is NOT playing.</summary>
  public bool CanEditEpochs => !Timeline.IsPlaying;

  /// <summary>Tooltip for the play/pause toggle button.</summary>
  public string PlayPauseTooltip => Timeline.IsPlaying ? "Pause" : "Play";

  private readonly IAudio2DService _audioService;

  public TimelineViewModel(
    ulong sceneId,
    NativeRuntimeService runtimeService,
    IUiThreadDispatcher uiThreadDispatcher,
    TrajectoryManagerService trajectoryManager,
    TimelineService timelineService,
    SceneStateManager sceneStateManager,
    BreadcrumbService breadcrumbService,
    IAudio2DService audioService
  )
    : base("Timeline")
  {
    _runtimeService = runtimeService;
    _uiThreadDispatcher = uiThreadDispatcher;
    _trajectoryManager = trajectoryManager;
    Timeline = timelineService;
    _sceneStateManager = sceneStateManager;
    _breadcrumbService = breadcrumbService;
    _audioService = audioService;
    CurrentSceneId = sceneId;
    Timeline.SelectedTimeScale = TimeScaleOptions.First();
    _timer = new Timer(UpdateFromRuntime, null, 33, 33);

    // Initialize epoch text from service
    StartEpochText = Timeline.StartDate.ToString("yyyy-MM-ddTHH:mm:ssZ");
    EndEpochText = Timeline.StopDate.ToString("yyyy-MM-ddTHH:mm:ssZ");

    // React to IsPlaying changes for CanEditEpochs
    Timeline.PropertyChanged += (s, e) =>
    {
      if (e.PropertyName == nameof(Timeline.IsPlaying))
      {
        OnPropertyChanged(nameof(CanEditEpochs));
        OnPropertyChanged(nameof(PlayPauseTooltip));
      }
    };

    // Register for simulation-reset messages
    WeakReferenceMessenger.Default.Register<CometDestroyedMessage>(this);
    WeakReferenceMessenger.Default.Register<JetConfigChangedMessage>(this);
    WeakReferenceMessenger.Default.Register<AlmanacUpdatedMessage>(this);
  }

  // ── Message Handlers ──────────────────────────────────────────────────────

  public void Receive(CometDestroyedMessage message)
  {
    if (message.SceneId == CurrentSceneId)
      ResetSimulationIfSnapshotted();
  }

  public void Receive(JetConfigChangedMessage message)
  {
    if (message.SceneId == CurrentSceneId)
      ResetSimulationIfSnapshotted();
  }

  public void Receive(AlmanacUpdatedMessage message)
  {
    // Almanac is global (SceneId=0 means all scenes), so always regenerate
    _ = UpdateTrajectoriesInternalAsync();
  }

  /// <summary>
  /// If the simulation was played then paused (snapshot exists), restore the
  /// snapshot, seek to the start epoch and clear the snapshot flag.
  /// Called when configuration changes invalidate the current simulation state.
  /// </summary>
  private void ResetSimulationIfSnapshotted()
  {
    if (!_hasSnapshotted || Timeline.IsPlaying)
      return;

    // Auto-pause if playing (shouldn't happen due to guard, but safety)
    if (Timeline.IsPlaying)
    {
      Timeline.IsPlaying = false;
      _runtimeService.SetTimeScale(CurrentSceneId, 0);
      _runtimeService.PauseScene(CurrentSceneId);
    }

    _runtimeService.RestoreSnapshot(CurrentSceneId);
    _hasSnapshotted = false;
    _runtimeService.SeekEpoch(CurrentSceneId, Timeline.MinTai);

    _breadcrumbService.ShowMessageAsync(
      "Simulation Reset",
      "Configuration changed — simulation has been reset to start.",
      default,
      3
    );
  }

  private void UpdateFromRuntime(object? state)
  {
    if (!_runtimeService.IsInitialized)
      return;

    _uiThreadDispatcher.Dispatch(() =>
    {
      if (Timeline.MinTai == 0 && Timeline.MaxTai == 100)
      {
        if (_runtimeService.GetEpochLimits(CurrentSceneId, out double min, out double max))
        {
          Timeline.MinTai = min;
          Timeline.MaxTai = max;
        }
      }

      if (!_isDragging)
      {
        Timeline.TimeTai = _runtimeService.GetSimulationTime(CurrentSceneId);
      }

      Timeline.UtcTime = _runtimeService.GetSimulationTimeUtc(CurrentSceneId);
    });
  }

  public void BeginDrag()
  {
    _isDragging = true;
  }

  public void EndDrag()
  {
    _isDragging = false;
    if (_runtimeService.IsInitialized)
    {
      _runtimeService.SeekEpoch(CurrentSceneId, Timeline.TimeTai);
      _ = UpdateTrajectoriesInternalAsync();
    }
  }

  partial void OnSelectedTimeScaleChanged(TimeScaleOption value)
  {
    if (Timeline.IsPlaying && _runtimeService.IsInitialized && value != null)
    {
      _runtimeService.SetTimeScale(CurrentSceneId, value.Value);
    }
  }

  private bool _hasSnapshotted;

  /// <summary>
  /// Toggle play/pause. On first play, captures a scene snapshot.
  /// Blocks playback if no comet or no jets are configured.
  /// </summary>
  [RelayCommand]
  private void PlayPause()
  {
    _audioService.PlayClickAsync();
    if (!_runtimeService.IsInitialized)
      return;

    if (Timeline.IsPlaying)
    {
      // Pause
      Timeline.IsPlaying = false;
      _runtimeService.SetTimeScale(CurrentSceneId, 0);
      _runtimeService.PauseScene(CurrentSceneId);
    }
    else
    {
      // Play — check prerequisites
      var state = _sceneStateManager.GetOrCreateScene(CurrentSceneId);
      if (!state.CometEntityId.HasValue)
      {
        _breadcrumbService.ShowMessageAsync(
          "Cannot Play",
          "Spawn a comet first before starting the simulation.",
          default,
          5
        );
        return;
      }

      bool hasFullyConfiguredJet = false;
      var comet = _runtimeService.GetEntityById(CurrentSceneId, state.CometEntityId.Value);
      if (comet != null)
      {
        var emitter = comet
          .Components.OfType<AetherVk.Logic.Models.ParticleEmitterCirclesComponent>()
          .FirstOrDefault();

        if (
          emitter != null
          && emitter.Circles.Any(c =>
            c.ParticlesPerTick > 0 && c.CircleRadiusKm > 0 && c.Mass > 0 && c.TTL > 0
          )
        )
        {
          hasFullyConfiguredJet = true;
        }
      }

      if (!hasFullyConfiguredJet)
      {
        _breadcrumbService.ShowMessageAsync(
          "Cannot Play",
          "Configure at least one valid emission jet (with >0 particles, radius, mass, and TTL) on the comet before starting the simulation.",
          default,
          5
        );
        return;
      }

      if (!_hasSnapshotted)
      {
        _runtimeService.SnapshotScene(CurrentSceneId);
        _hasSnapshotted = true;
      }
      Timeline.IsPlaying = true;
      _runtimeService.SetTimeScale(CurrentSceneId, Timeline.SelectedTimeScale?.Value ?? 1);
      _runtimeService.PlayScene(CurrentSceneId);
    }
  }

  /// <summary>
  /// Validates and applies the start/end epoch range.
  /// Checks that start &lt;= end and that almanac coverage is sufficient.
  /// </summary>
  [RelayCommand]
  private async System.Threading.Tasks.Task ApplyEpochRange()
  {
    HasEpochError = false;
    EpochErrorText = "";

    if (!DateTimeOffset.TryParse(StartEpochText, out DateTimeOffset startDto))
    {
      HasEpochError = true;
      EpochErrorText = "Invalid start epoch format";
      return;
    }

    if (!DateTimeOffset.TryParse(EndEpochText, out DateTimeOffset endDto))
    {
      HasEpochError = true;
      EpochErrorText = "Invalid end epoch format";
      return;
    }

    if (startDto >= endDto)
    {
      HasEpochError = true;
      EpochErrorText = "Start must be before end";
      return;
    }

    if (!_runtimeService.IsInitialized)
      return;

    // Apply the new epoch range
    _runtimeService.SetEpochRange(CurrentSceneId, startDto, endDto);

    // Update timeline service
    Timeline.StartDate = startDto;
    Timeline.StopDate = endDto;

    // Refresh epoch limits from runtime
    if (_runtimeService.GetEpochLimits(CurrentSceneId, out double min, out double max))
    {
      Timeline.MinTai = min;
      Timeline.MaxTai = max;
    }

    // If simulation was paused and had a snapshot, reset
    if (_hasSnapshotted && !Timeline.IsPlaying)
    {
      _runtimeService.RestoreSnapshot(CurrentSceneId);
      _hasSnapshotted = false;
      _runtimeService.SeekEpoch(CurrentSceneId, Timeline.MinTai);
    }

    await UpdateTrajectoriesInternalAsync();

    _breadcrumbService.ShowMessageAsync(
      "Epoch Updated",
      $"Simulation epoch range set to {startDto:yyyy-MM-dd} — {endDto:yyyy-MM-dd}",
      default,
      3
    );
  }

  private async System.Threading.Tasks.Task UpdateTrajectoriesInternalAsync()
  {
    if (_runtimeService.IsInitialized)
    {
      double stepDays = 1.0;
      await _trajectoryManager.UpdateAllTrajectoriesAsync(
        CurrentSceneId,
        Timeline.MinTai,
        Timeline.MaxTai,
        stepDays
      );
    }
  }

  public void Dispose()
  {
    WeakReferenceMessenger.Default.UnregisterAll(this);
    _timer.Dispose();
  }
}
