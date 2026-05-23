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
  private readonly Timer _timer;
  private bool _isDragging;

  [ObservableProperty]
  private ulong _currentSceneId;

  [ObservableProperty]
  private string _utcTime = "Loading...";

  [ObservableProperty]
  private double _timeTai;

  [ObservableProperty]
  private double _minTai = 0;

  [ObservableProperty]
  private double _maxTai = 100;

  [ObservableProperty]
  private bool _isPlaying;

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
    TrajectoryManagerService trajectoryManager
  )
    : base("Timeline")
  {
    _runtimeService = runtimeService;
    _uiThreadDispatcher = uiThreadDispatcher;
    _trajectoryManager = trajectoryManager;
    CurrentSceneId = sceneId;
    SelectedTimeScale = TimeScaleOptions.First();
    _timer = new Timer(UpdateFromRuntime, null, 33, 33);
  }

  private void UpdateFromRuntime(object? state)
  {
    if (!_runtimeService.IsInitialized)
      return;

    _uiThreadDispatcher.Dispatch(() =>
    {
      if (MinTai == 0 && MaxTai == 100)
      {
        if (_runtimeService.GetEpochLimits(CurrentSceneId, out double min, out double max))
        {
          MinTai = min;
          MaxTai = max;
        }
      }

      if (!_isDragging)
      {
        TimeTai = _runtimeService.GetSimulationTime(CurrentSceneId);
      }

      UtcTime = _runtimeService.GetSimulationTimeUtc(CurrentSceneId);
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
      _runtimeService.SetSimulationTime(CurrentSceneId, TimeTai);
    }
  }

  partial void OnTimeTaiChanged(double value)
  {
    if (_isDragging && _runtimeService.IsInitialized)
    {
      _runtimeService.SetSimulationTime(CurrentSceneId, value);
    }
  }

  partial void OnSelectedTimeScaleChanged(TimeScaleOption value)
  {
    if (IsPlaying && _runtimeService.IsInitialized && value != null)
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
      if (!_hasSnapshotted)
      {
        _runtimeService.SnapshotScene(CurrentSceneId);
        _hasSnapshotted = true;
      }
      IsPlaying = true;
      _runtimeService.SetTimeScale(CurrentSceneId, SelectedTimeScale?.Value ?? 1);
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
      IsPlaying = false;
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
      IsPlaying = false;
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
      await _trajectoryManager.UpdateAllTrajectoriesAsync(CurrentSceneId, MinTai, MaxTai, stepDays);
    }
  }

  public void Dispose()
  {
    _timer.Dispose();
  }
}
