using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The jets ignite a lead time before the committed start (reviewer report 2026-10-29: "the model
/// appears only if you go to the end"; igniting at the start left an empty tail at play).
/// </summary>
public class DustLeadTimeTests
{
  private const long UsPerDay = 86_400L * 1_000_000L;

  [Fact]
  public void Ignition_IsTheCommittedStartMinusTheLead_InTdbMicroseconds()
  {
    // 2021-12-12T00:00 TDB = 692 539 200 s after J2000 (the reviewer's committed start)
    var range = new TimeRange(0, 692_539_200_000_000_000UL, 0, 693_921_600_000_000_000UL);
    Assert.Equal(692_539_200_000_000L - 30 * UsPerDay, TimelineService.IgnitionUs(range, 30));
    Assert.Equal(692_539_200_000_000L, TimelineService.IgnitionUs(range, 0));
    Assert.Equal(692_539_200_000_000L - UsPerDay / 2, TimelineService.IgnitionUs(range, 0.5));
    // another century: centuries count 36 525 days
    var later = new TimeRange(1, 0, 1, 1);
    Assert.Equal(36_525L * UsPerDay - UsPerDay, TimelineService.IgnitionUs(later, 1));
  }

  [Fact]
  public void WithoutACommittedComet_TheJetIgnitesAtTheSceneClock()
  {
    Assert.Equal(long.MaxValue, TimelineService.IgnitionUs(null, 30));
  }

  [Theory]
  [InlineData(45.0, 30.0)]
  [InlineData(-1.0, 0.0)]
  [InlineData(double.NaN, 30.0)]
  [InlineData(12.5, 12.5)]
  public void TheLead_IsClampedToOneTtl(double set, double expected)
  {
    var rt = new Moq.Mock<INativeRuntimeService>();
    var sched = new Mocks.TestSchedulerProvider();
    var tl = new TimelineService(rt.Object, sched, new CometConfigService(rt.Object, sched), new BreadcrumbService());
    tl.DustLeadDays = set;
    Assert.Equal(expected, tl.DustLeadDays);
  }
}
