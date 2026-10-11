using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// Sky image overlay (WP5): the <c>CSkyImageInfo</c> layout, the shared maths (ecliptic direction,
/// TDB parts) and <see cref="SkyImageOverlayViewModel"/> (offers enabled from the header facts,
/// the missing-header breadcrumb, the display parameters pushed to the runtime).
/// </summary>
public class SkyImageOverlayViewModelTests
{
  private sealed class Setup
  {
    public Mock<INativeRuntimeService> Runtime { get; } = new();
    public List<(double Lat, double Lon, double? ElevM)> Sites { get; } = [];
    public List<Vector3d> Locks { get; } = [];
    public bool LockAccepted { get; set; } = true;
    public SkyImageOverlayViewModel Vm { get; }

    public Setup()
    {
      var translation = new Mock<ITranslationService>();
      translation
        .Setup(t => t.CultureChanged)
        .Returns(System.Reactive.Linq.Observable.Never<System.Globalization.CultureInfo>());
      // the key itself: the tests read which string is shown
      translation.Setup(t => t.GetString(It.IsAny<string>())).Returns<string>(k => k);
      var schedulers = new Mock<ISchedulerProvider>();
      schedulers
        .Setup(s => s.MainThread)
        .Returns(System.Reactive.Concurrency.ImmediateScheduler.Instance);
      schedulers
        .Setup(s => s.Background)
        .Returns(System.Reactive.Concurrency.ImmediateScheduler.Instance);
      var dialog = new Mock<IFileDialogService>();
      dialog
        .Setup(d => d.ShowOpenFileDialogAsync(It.IsAny<string>(), It.IsAny<string[]?>()))
        .ReturnsAsync("/data/m42.fits");
      Vm = new SkyImageOverlayViewModel(
        Runtime.Object,
        dialog.Object,
        translation.Object,
        schedulers.Object,
        (lat, lon, elev) => Sites.Add((lat, lon, elev)),
        d =>
        {
          Locks.Add(d);
          return LockAccepted;
        });
    }
  }

  private static SkyImageInfo FullInfo(uint extraFlags = 0) => SkyImageInfo.FromNative(new CSkyImageInfo
  {
    HasWcs = 1,
    Nx = 4096,
    Ny = 4096,
    DateObsUs = 788_961_669_184_000,
    ExptimeS = 120.0,
    SiteLatDeg = 45.5,
    SiteLonDeg = -110.5,
    SiteElevM = 1234.5,
    CrvalRaDeg = 83.8,
    CrvalDecDeg = -5.4,
    Flags = SkyImageInfo.FlagDateObs | SkyImageInfo.FlagExptime | SkyImageInfo.FlagSiteLat
      | SkyImageInfo.FlagSiteLon | SkyImageInfo.FlagSiteElev | extraFlags,
  });

  [Fact]
  public void CSkyImageInfo_matches_the_native_layout()
  {
    Assert.Equal(80, Marshal.SizeOf<CSkyImageInfo>());
    Assert.Equal(16, (int)Marshal.OffsetOf<CSkyImageInfo>(nameof(CSkyImageInfo.DateObsUs)));
    Assert.Equal(56, (int)Marshal.OffsetOf<CSkyImageInfo>(nameof(CSkyImageInfo.CrvalRaDeg)));
    Assert.Equal(72, (int)Marshal.OffsetOf<CSkyImageInfo>(nameof(CSkyImageInfo.Flags)));
  }

  [Fact]
  public void FromNative_keeps_only_the_flagged_header_fields()
  {
    var info = SkyImageInfo.FromNative(new CSkyImageInfo
    {
      HasWcs = 1,
      DateObsUs = 42,
      ExptimeS = 3.0,
      SiteLatDeg = 10.0,
      Flags = SkyImageInfo.FlagDateObs | SkyImageInfo.FlagRaster,
    });
    Assert.True(info.HasWcs);
    Assert.Equal(42, info.DateObsTdbUs);
    Assert.Null(info.ExptimeS);
    Assert.Null(info.SiteLatDeg);
    Assert.True(info.IsRaster);
    Assert.False(info.SipIgnored);
  }

  [Fact]
  public void Tdb_parts_split_by_julian_century_with_non_negative_nanoseconds()
  {
    Assert.Equal(((short)0, 0UL), SkyImageMath.TdbMicrosToParts(0));
    Assert.Equal(((short)-1, (ulong)(SkyImageMath.NanosecondsPerCentury - 1000)), SkyImageMath.TdbMicrosToParts(-1));
    long oneCenturyUs = SkyImageMath.NanosecondsPerCentury / 1000;
    Assert.Equal(((short)1, 5000UL), SkyImageMath.TdbMicrosToParts(oneCenturyUs + 5));
    Assert.Equal(1_000_060_000_000L, SkyImageMath.MidExposureTdbUs(1_000_000_000_000L, 120.0));
    Assert.Equal(7L, SkyImageMath.MidExposureTdbUs(7L, null));
  }

  [Fact]
  public void Ecliptic_direction_fixes_the_equinox_and_maps_the_ecliptic_pole_to_z()
  {
    var x = SkyImageMath.EclipticDirection(0.0, 0.0);
    Assert.Equal(1.0, x.X, 12);
    double eps = SkyImageMath.ObliquityRad * 180.0 / Math.PI;
    Assert.Equal(23.4392911, eps, 6);
    var pole = SkyImageMath.EclipticDirection(270.0, 90.0 - eps);
    Assert.Equal(1.0, pole.Z, 12);
  }

  [Fact]
  public async Task Missing_header_fields_disable_their_offers_and_show_a_breadcrumb()
  {
    var s = new Setup();
    var bare = SkyImageInfo.FromNative(new CSkyImageInfo { HasWcs = 0, Nx = 10, Ny = 10, Flags = SkyImageInfo.FlagRaster });
    s.Runtime.Setup(r => r.LoadSkyImage("/data/m42.png", null)).Returns(bare);
    await s.Vm.LoadPathAsync("/data/m42.png");

    Assert.True(s.Vm.HasImage);
    Assert.False(s.Vm.SeekToDateObsCommand.CanExecute(null));
    Assert.False(s.Vm.UseSiteFromHeaderCommand.CanExecute(null));
    Assert.False(s.Vm.LockToImageCommand.CanExecute(null));
    Assert.Equal("Panels_SkyImage_MissingDate", s.Vm.SeekToolTip);
    Assert.Equal("Panels_SkyImage_MissingSite", s.Vm.SiteToolTip);
    Assert.Equal("Panels_SkyImage_MissingWcs", s.Vm.LockToolTip);
    Assert.True(s.Vm.HasMissingHeader);
    Assert.Equal(
      "Panels_SkyImage_MissingWcs\nPanels_SkyImage_MissingDate\nPanels_SkyImage_MissingSite",
      s.Vm.MissingHeaderMessage);
    // no WCS: nothing drawn, no parameters pushed
    s.Vm.Opacity = 0.3;
    s.Runtime.Verify(r => r.SetSkyImageParams(It.IsAny<float>(), It.IsAny<float>(), It.IsAny<bool>()), Times.Never);
  }

  [Fact]
  public async Task A_full_header_enables_seek_site_and_lock()
  {
    var s = new Setup();
    s.Runtime.Setup(r => r.LoadSkyImage("/data/m42.fits", null)).Returns(FullInfo());
    s.Runtime.Setup(r => r.SeekEpochSync(It.IsAny<short>(), It.IsAny<ulong>())).Returns(true);
    await s.Vm.LoadImageCommand.ExecuteAsync(null);

    Assert.Equal("m42.fits", s.Vm.ImageName);
    Assert.False(s.Vm.HasMissingHeader);
    Assert.Equal(SkyImageOverlayViewModel.DefaultFitsStretch, s.Vm.Stretch);
    s.Runtime.Verify(r => r.SetSkyImageParams(0.7f, (float)SkyImageOverlayViewModel.DefaultFitsStretch, true), Times.Once);

    // seek: DATE-OBS + EXPTIME / 2
    s.Vm.SeekToDateObsCommand.Execute(null);
    var (c, ns) = SkyImageMath.TdbMicrosToParts(788_961_669_184_000 + 60_000_000);
    s.Runtime.Verify(r => r.SeekEpochSync(c, ns), Times.Once);

    s.Vm.UseSiteFromHeaderCommand.Execute(null);
    Assert.Equal(new List<(double, double, double?)> { (45.5, -110.5, 1234.5) }, s.Sites);

    s.Vm.LockToImageCommand.Execute(null);
    var want = SkyImageMath.EclipticDirection(83.8, -5.4);
    Assert.Single(s.Locks);
    Assert.Equal(want.X, s.Locks[0].X, 15);
    Assert.Equal(want.Y, s.Locks[0].Y, 15);
    Assert.Equal(want.Z, s.Locks[0].Z, 15);
    Assert.DoesNotContain("LockFailed", s.Vm.StatusMessage);

    s.LockAccepted = false;
    s.Vm.LockToImageCommand.Execute(null);
    Assert.Equal("Panels_SkyImage_LockFailed", s.Vm.StatusMessage);
  }

  [Fact]
  public async Task Display_parameters_are_pushed_and_remove_clears_the_overlay()
  {
    var s = new Setup();
    s.Runtime
      .Setup(r => r.LoadSkyImage("/data/m42.jpg", null))
      .Returns(FullInfo(SkyImageInfo.FlagRaster | SkyImageInfo.FlagSipIgnored));
    s.Runtime.Setup(r => r.RemoveSkyImage()).Returns(true);
    await s.Vm.LoadPathAsync("/data/m42.jpg");
    // PNG / JPEG are already stretched: linear
    Assert.Equal(0.0, s.Vm.Stretch);
    Assert.Contains("Panels_SkyImage_SipIgnored", s.Vm.StatusMessage);

    s.Vm.Opacity = 1.5;
    s.Runtime.Verify(r => r.SetSkyImageParams(1.0f, 0.0f, true), Times.AtLeastOnce);
    s.Vm.IsOverlayVisible = false;
    s.Runtime.Verify(r => r.SetSkyImageParams(1.0f, 0.0f, false), Times.Once);

    Assert.True(s.Vm.RemoveImageCommand.CanExecute(null));
    s.Vm.RemoveImageCommand.Execute(null);
    s.Runtime.Verify(r => r.RemoveSkyImage(), Times.Once);
    Assert.False(s.Vm.HasImage);
    Assert.False(s.Vm.RemoveImageCommand.CanExecute(null));
    Assert.False(s.Vm.LockToImageCommand.CanExecute(null));
  }

  [Fact]
  public async Task A_failed_load_keeps_no_image_and_reports_it()
  {
    var s = new Setup();
    s.Runtime.Setup(r => r.LoadSkyImage(It.IsAny<string>(), It.IsAny<string?>())).Returns((SkyImageInfo?)null);
    await s.Vm.LoadPathAsync("/data/broken.fits");
    Assert.False(s.Vm.HasImage);
    Assert.Equal("Panels_SkyImage_StatusFailed: broken.fits", s.Vm.StatusMessage);
    Assert.True(s.Vm.LoadImageCommand.CanExecute(null));
  }
}
