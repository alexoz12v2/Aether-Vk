using System;
using AetherVk.Logic.Services;
using AetherVk.Logic.Tests.Mocks;
using AetherVk.Logic.ViewModels.Debug;
using Moq;
using Xunit;
using Microsoft.Reactive.Testing;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The viewport's ephemeris table (reviewer docx item 15): <see cref="CameraMatrixDebugViewModel"/>
/// polls <see cref="INativeRuntimeService.CometEphemeris"/> and shows the rows Horizons-style, "—"
/// without a committed comet, with the cartesian camera position under a collapsed Debug expander.
/// </summary>
public sealed class CometEphemerisPanelTests
{
  private const double Deg = Math.PI / 180.0;

  /// <summary>67P-like values: RA 07h 51m 23.4s, Dec +21° 05′ 12″, r 1.27314, Δ 0.44021 AU.</summary>
  private static CometEphemeris Sample() => new(
    new DateTimeOffset(2021, 12, 27, 0, 0, 0, TimeSpan.Zero),
    RaRad: (7.0 + 51.0 / 60.0 + 23.4 / 3600.0) * 15.0 * Deg,
    DecRad: (21.0 + 5.0 / 60.0 + 12.0 / 3600.0) * Deg,
    RAu: 1.27314,
    DeltaAu: 0.44021,
    PhaseRad: 48.23 * Deg,
    ElongationRad: 112.34 * Deg,
    SunPaRad: 287.61 * Deg,
    LightTimeS: 219.6);

  private static (CameraMatrixDebugViewModel Vm, TestSchedulerProvider Schedulers) Panel(CometEphemeris? ephemeris)
  {
    var runtime = new Mock<INativeRuntimeService>();
    runtime.Setup(r => r.CometEphemeris()).Returns(ephemeris);
    var schedulers = new TestSchedulerProvider();
    var vm = new CameraMatrixDebugViewModel(runtime.Object, schedulers);
    return (vm, schedulers);
  }

  private static void Poll(TestSchedulerProvider schedulers)
  {
    schedulers.Background.AdvanceBy(TimeSpan.FromMilliseconds(500).Ticks);
    schedulers.MainThread.AdvanceBy(1);
  }

  [Fact]
  public void WithoutACommittedComet_EveryRowIsADash()
  {
    var (vm, schedulers) = Panel(null);
    using (vm)
    {
      Poll(schedulers);
      foreach (var row in new[] { vm.EphemerisEpoch, vm.EphemerisRa, vm.EphemerisDec, vm.EphemerisR,
                 vm.EphemerisDelta, vm.EphemerisPhase, vm.EphemerisElongation, vm.EphemerisSunPa })
        Assert.Equal(CameraMatrixDebugViewModel.NoValue, row);
    }
  }

  [Fact]
  public void TheDebugExpander_IsCollapsedByDefault()
  {
    var (vm, _) = Panel(null);
    using (vm)
      Assert.False(vm.IsDebugExpanded);
  }

  [Fact]
  public void WithACommittedComet_TheRowsShowTheEphemeris()
  {
    var (vm, schedulers) = Panel(Sample());
    using (vm)
    {
      Poll(schedulers);
      Assert.Equal("2021-12-27 00:00 UTC", vm.EphemerisEpoch);
      Assert.Equal("07h 51m 23.4s", vm.EphemerisRa);
      Assert.Equal("+21° 05′ 12″", vm.EphemerisDec);
      Assert.Equal("1.2731 AU", vm.EphemerisR);
      Assert.Equal("0.4402 AU", vm.EphemerisDelta);
      Assert.Equal("48.2°", vm.EphemerisPhase);
      Assert.Equal("112.3°", vm.EphemerisElongation);
      Assert.Equal("287.6°", vm.EphemerisSunPa);
    }
  }

  [Fact]
  public void FormatEphemeris_NullIsEmpty_AndTheSampleIsHorizonsStyle()
  {
    Assert.Same(CameraMatrixDebugViewModel.CometEphemerisText.Empty, CameraMatrixDebugViewModel.FormatEphemeris(null));
    var t = CameraMatrixDebugViewModel.FormatEphemeris(Sample());
    Assert.Equal("2021-12-27 00:00 UTC", t.Epoch);
    Assert.Equal("07h 51m 23.4s", t.Ra);
    Assert.Equal("+21° 05′ 12″", t.Dec);
    Assert.Equal("1.2731 AU", t.R);
    Assert.Equal("0.4402 AU", t.Delta);
    Assert.Equal("48.2°", t.Phase);
    Assert.Equal("112.3°", t.Elongation);
    Assert.Equal("287.6°", t.SunPa);
  }

  [Fact]
  public void FormatRightAscension_WrapsAndCarries()
  {
    Assert.Equal("00h 00m 00.0s", CameraMatrixDebugViewModel.FormatRightAscension(0.0));
    // 23h 59m 59.97s rounds up and wraps to 0h
    Assert.Equal("00h 00m 00.0s", CameraMatrixDebugViewModel.FormatRightAscension(2.0 * Math.PI - 0.03 / 3600.0 * 15.0 * Deg));
    // a negative angle is the same direction as its positive wrap
    Assert.Equal("18h 00m 00.0s", CameraMatrixDebugViewModel.FormatRightAscension(-Math.PI / 2.0));
    Assert.Equal("12h 30m 00.0s", CameraMatrixDebugViewModel.FormatRightAscension(12.5 * 15.0 * Deg));
  }

  [Fact]
  public void FormatDeclination_IsSignedSexagesimal()
  {
    Assert.Equal("+00° 00′ 00″", CameraMatrixDebugViewModel.FormatDeclination(0.0));
    Assert.Equal("-05° 30′ 00″", CameraMatrixDebugViewModel.FormatDeclination(-5.5 * Deg));
    Assert.Equal("+90° 00′ 00″", CameraMatrixDebugViewModel.FormatDeclination(Math.PI / 2.0));
    // 59.6″ rounds to the next minute
    Assert.Equal("-12° 35′ 00″", CameraMatrixDebugViewModel.FormatDeclination(-(12.0 + 34.0 / 60.0 + 59.6 / 3600.0) * Deg));
  }
}
