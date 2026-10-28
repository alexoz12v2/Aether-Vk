using System;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Xunit;

namespace AetherVk.Logic.Tests;

/// Feedback #16 / #20: the overlay's frame label and the timeline's model range.
public class ViewModesTests
{
  [Fact]
  public void FrameLabel_NamesTheFrameOfEachMode()
  {
    Assert.Contains("Ecliptic", ViewportOverlayViewModel.FrameLabelFor(EarthObserverState.UpZenith, ObserverUp.Ecliptic));
    Assert.Contains("body frame", ViewportOverlayViewModel.FrameLabelFor(EarthObserverState.CometOrbiting, ObserverUp.CelestialNorth));
    Assert.Contains("celestial north", ViewportOverlayViewModel.FrameLabelFor(EarthObserverState.EarthPositioning, ObserverUp.CelestialNorth));
    Assert.Contains("ecliptic north", ViewportOverlayViewModel.FrameLabelFor(EarthObserverState.EarthPositioning, ObserverUp.Ecliptic));
  }

  [Fact]
  public void ModelRange_IsNRotationsEndingAtTheImage_AtLeastTheMinimum()
  {
    var end = new DateTimeOffset(2021, 12, 27, 0, 14, 47, TimeSpan.Zero);

    var r = TimelineTabViewModel.ModelRange(end, 3, 12.4);
    Assert.NotNull(r);
    Assert.Equal(end, r!.Value.End);
    Assert.Equal(end - TimeSpan.FromHours(37.2), r.Value.Start);

    // shorter than the timeline accepts: widened to the minimum
    var shortRange = TimelineTabViewModel.ModelRange(end, 1, 12.4);
    Assert.Equal(end - TimeSpan.FromDays(TimelineTabViewModel.MinRangeDays), shortRange!.Value.Start);

    Assert.Null(TimelineTabViewModel.ModelRange(end, 3, 0));
    Assert.Null(TimelineTabViewModel.ModelRange(end, 0, 12.4));
  }
}
