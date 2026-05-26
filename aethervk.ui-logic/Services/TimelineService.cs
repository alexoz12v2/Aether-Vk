using System;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Messaging;
using AetherVk.Logic.ViewModels;

namespace AetherVk.Logic.Services;

public partial class TimelineService : ObservableObject
{
  [ObservableProperty]
  private double _minTai = 0;

  [ObservableProperty]
  private double _maxTai = 100;

  [ObservableProperty]
  private double _timeTai;

  [ObservableProperty]
  private string _utcTime = "Loading...";

  [ObservableProperty]
  private bool _isPlaying;

  [ObservableProperty]
  private TimeScaleOption? _selectedTimeScale;

  [ObservableProperty]
  private DateTimeOffset _startDate;

  [ObservableProperty]
  private DateTimeOffset _stopDate;

  public TimelineService()
  {
    // Default to last 10 years for startup
    StopDate = DateTimeOffset.UtcNow;
    StartDate = StopDate.AddYears(-10);
  }

  // Called when the physics runtime returns valid epoch limits
  public void UpdateEpochLimits(double min, double max, DateTimeOffset start, DateTimeOffset stop)
  {
    MinTai = min;
    MaxTai = max;
    StartDate = start;
    StopDate = stop;
  }
}
