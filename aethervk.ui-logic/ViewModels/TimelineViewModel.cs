using System;
using System.Threading;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

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

  [RelayCommand]
  private void SetSpeed(string speedStr)
  {
    if (uint.TryParse(speedStr, out uint speed) && _runtimeService.IsInitialized)
    {
      _runtimeService.SetTimeScale(CurrentSceneId, speed);
      if (speed == 0)
        _runtimeService.PauseScene(CurrentSceneId);
      else
        _runtimeService.PlayScene(CurrentSceneId);
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
