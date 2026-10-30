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
    Assert.StartsWith("Orbit-plane frame", ViewportOverlayViewModel.FrameLabelFor(EarthObserverState.UpZenith, ObserverUp.Ecliptic));
    Assert.StartsWith("Body-fixed frame", ViewportOverlayViewModel.FrameLabelFor(EarthObserverState.CometOrbiting, ObserverUp.CelestialNorth));
    Assert.StartsWith("From Earth, celestial north up", ViewportOverlayViewModel.FrameLabelFor(EarthObserverState.EarthPositioning, ObserverUp.CelestialNorth));
    Assert.StartsWith("From Earth, ecliptic north up", ViewportOverlayViewModel.FrameLabelFor(EarthObserverState.EarthPositioning, ObserverUp.Ecliptic));
  }

  /// Feedback #17: the nucleus-from-Earth view is the comet orbit posed by the core on the
  /// Earth → comet line: its label says "from Earth", not the body-fixed frame, and keeps the epoch.
  [Fact]
  public void FrameLabel_NucleusFromEarth_SaysFromEarth()
  {
    const string epoch = "2021-12-27 00:14:47 UTC";
    string label = ViewportOverlayViewModel.FrameLabelFor(EarthObserverState.CometOrbiting, ObserverUp.Ecliptic, epoch, nucleusFromEarth: true);
    Assert.StartsWith("Nucleus from Earth", label);
    Assert.Contains("from Earth", label);
    Assert.EndsWith(" · " + epoch, label);
    // the flag only applies to the comet orbit; off, the orbit is the body-fixed frame
    Assert.StartsWith("Body-fixed frame", ViewportOverlayViewModel.FrameLabelFor(EarthObserverState.CometOrbiting, ObserverUp.Ecliptic, epoch, nucleusFromEarth: false));
    Assert.StartsWith("Orbit-plane frame", ViewportOverlayViewModel.FrameLabelFor(EarthObserverState.UpZenith, ObserverUp.Ecliptic, epoch, nucleusFromEarth: true));
  }

  /// Feedback #16: the label states at which date the view is drawn (the epoch the overlay shows).
  [Theory]
  [InlineData(EarthObserverState.UpZenith, ObserverUp.Ecliptic, "Orbit-plane frame")]
  [InlineData(EarthObserverState.CometOrbiting, ObserverUp.Ecliptic, "Body-fixed frame")]
  [InlineData(EarthObserverState.EarthPositioning, ObserverUp.CelestialNorth, "From Earth, celestial north up")]
  [InlineData(EarthObserverState.EarthPositioning, ObserverUp.Ecliptic, "From Earth, ecliptic north up")]
  public void FrameLabel_EndsWithTheShownEpoch(EarthObserverState mode, ObserverUp up, string frame)
  {
    const string epoch = "2021-12-27 00:14:47 UTC";
    string label = ViewportOverlayViewModel.FrameLabelFor(mode, up, epoch);
    Assert.StartsWith(frame, label);
    Assert.EndsWith(" · " + epoch, label);
    // no epoch yet: the frame alone, no dangling separator
    Assert.DoesNotContain("·", ViewportOverlayViewModel.FrameLabelFor(mode, up, ""));
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
