using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

public partial class TimelineTabViewModel : TabItemViewModel
{
  public TimelineTabViewModel() : base("Timeline")
  {
    Icon = "⏱"; // stopwatch — U+23F1
  }
}
