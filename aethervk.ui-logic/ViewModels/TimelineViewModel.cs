using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

public class TimeScaleOption
{
    public string DisplayName { get; init; } = "";
    public uint Value { get; init; }
}

public partial class TimelineViewModel : TabItemViewModel, IDisposable
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

  public ObservableCollection<TimeScaleOption> TimeScaleOptions { get; } = new()
  {
      new TimeScaleOption { DisplayName = "1 Day/sec", Value = 1 },
      new TimeScaleOption { DisplayName = "1 Week/sec", Value = 2 },
      new TimeScaleOption { DisplayName = "1 Month/sec", Value = 3 },
  };

  [ObservableProperty]
  private TimeScaleOption _selectedTimeScale;

  public TimelineViewModel(
    ulong sceneId,
    NativeRuntimeService runtimeService,
    IUiThreadDispatcher uiThreadDispatcher,
    TrajectoryManagerService trajectoryManager,
    TimelineService timelineService,
    SceneStateManager sceneStateManager,
    BreadcrumbService breadcrumbService
  )
    : base("Timeline")
  {
    _runtimeService = runtimeService;
    _uiThreadDispatcher = uiThreadDispatcher;
    _trajectoryManager = trajectoryManager;
    Timeline = timelineService;
    _sceneStateManager = sceneStateManager;
    _breadcrumbService = breadcrumbService;
    CurrentSceneId = sceneId;
    Timeline.SelectedTimeScale = TimeScaleOptions.First();
    _timer = new Timer(UpdateFromRuntime, null, 33, 33);
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
      _runtimeService.SetSimulationTime(CurrentSceneId, Timeline.TimeTai);
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
  /// Initiates scene playback. 
  /// Automatically captures a snapshot of the simulation state if it's the first time playing, 
  /// sets the simulation timescale to 1, and pushes the PlayScene command down to the native logic thread.
  /// </summary>
  [RelayCommand]
  private void Play()
  {
    if (_runtimeService.IsInitialized)
    {
      var state = _sceneStateManager.GetOrCreateScene(CurrentSceneId);
      bool hasJets = false;
      if (state.CometEntityId.HasValue)
      {
          var comet = _runtimeService.GetEntityById(CurrentSceneId, state.CometEntityId.Value);
          if (comet != null)
          {
              var emitter = comet.Components.OfType<AetherVk.Logic.Models.ParticleEmitterCirclesComponent>().FirstOrDefault();
              if (emitter != null && emitter.Circles.Count > 0)
              {
                  hasJets = true;
              }
          }
      }

      if (!hasJets)
      {
          _breadcrumbService.ShowMessageAsync("Cannot Play", "Please add at least 1 jet to the comet before playing the simulation.", default, 5);
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
  /// Pauses scene playback.
  /// Zeroes out the timescale and dispatches a PauseScene command to the native logic thread.
  /// </summary>
  [RelayCommand]
  private void Pause()
  {
    if (_runtimeService.IsInitialized)
    {
      Timeline.IsPlaying = false;
      _runtimeService.SetTimeScale(CurrentSceneId, 0);
      _runtimeService.PauseScene(CurrentSceneId);
    }
  }

  /// <summary>
  /// Stops playback entirely and rewinds the simulation.
  /// Resets the time scale, pauses the logic engine, and gracefully restores 
  /// the simulation state from the initial snapshot if one exists.
  /// </summary>
  [RelayCommand]
  private void Stop()
  {
    if (_runtimeService.IsInitialized)
    {
      Timeline.IsPlaying = false;
      _runtimeService.SetTimeScale(CurrentSceneId, 0);
      _runtimeService.PauseScene(CurrentSceneId);
      if (_hasSnapshotted)
      {
        _runtimeService.RestoreSnapshot(CurrentSceneId);
      }
    }
  }



  [RelayCommand]
  private async System.Threading.Tasks.Task UpdateTrajectoriesAsync()
  {
    if (_runtimeService.IsInitialized)
    {
      double stepDays = 1.0;
      await _trajectoryManager.UpdateAllTrajectoriesAsync(CurrentSceneId, Timeline.MinTai, Timeline.MaxTai, stepDays);
    }
  }

  public void Dispose()
  {
    _timer.Dispose();
  }
}
