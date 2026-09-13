using System;
using System.Threading.Tasks;
using Avalonia.Headless.XUnit;
using Microsoft.Extensions.DependencyInjection;
using Xunit;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;

namespace AetherVk.AppTests.Integration;

public class JetAndTimelineTests
{
  [AvaloniaFact]
  public async Task Timeline_PlayPause_ResetsAndPlays()
  {
    var host = AetherVk.App.Host ?? throw new InvalidOperationException("DI Host not initialized.");
    await using var scope = host.Services.CreateAsyncScope();
    
    var timelineVm = scope.ServiceProvider.GetRequiredService<TimelineTabViewModel>();
    var timelineService = host.Services.GetRequiredService<TimelineService>();
    
    Assert.False(timelineVm.IsPlaying);
    
    timelineVm.PlayPauseCommand.Execute(null);
    Assert.True(timelineVm.IsPlaying);
    
    timelineVm.ResetCommand.Execute(null);
    Assert.False(timelineVm.IsPlaying);
  }

  [AvaloniaFact]
  public async Task JetVisibility_Toggles_State()
  {
    var host = AetherVk.App.Host ?? throw new InvalidOperationException("DI Host not initialized.");
    await using var scope = host.Services.CreateAsyncScope();
    
    var modelTab = scope.ServiceProvider.GetRequiredService<ModelTabViewModel>();
    
    // Create a new jet
    modelTab.AddJetCommand.Execute(null);
    Assert.Single(modelTab.Jets!);
  }
}
