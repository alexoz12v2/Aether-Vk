using System;
using System.Collections.Generic;
using System.Linq;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The nucleus pixel of a billboard: image px → desktop px through the billboard's scale and
/// rotation, and "Center on comet" putting it (not the image centre) on the comet.
/// </summary>
public class DesktopBillboardGeometryTests
{
  private static void Near(double want, double got, double tol = 1e-9) =>
    Assert.True(Math.Abs(want - got) <= tol, $"expected {want}, got {got}");

  private static void Near(PointD want, PointD got, double tol = 1e-9)
  {
    Near(want.X, got.X, tol);
    Near(want.Y, got.Y, tol);
  }

  /// A viewport that only reports the comet.
  private sealed class CometProbe(PointD comet) : IBillboardViewportProbe
  {
    public double? ArcsecPerPx => 1.0;
    public PointD? CometDesktopPx => comet;
    public double? CometDistanceKm => null;
    public double ViewportHeightPx => 1000;
    public double? VerticalFovRad => null;
    public bool IsOrthographic => false;
    public bool MatchField(double heightKm, double angleRad) => false;
    public bool IsEarthObserver => true;
    public bool EnterEarthObserver() => true;
    public void SetObserverSite(double latDeg, double lonDeg, double? elevM) { }
    public bool LockToDirection(Vector3d worldDirection) => true;
    public bool ViewUpIsEclipticSouth => false;
    public bool ViewUpIsCelestialNorth => true;
    public PointD? ViewportCenterDesktopPx => null;
    public ViewportFrame? ViewportFrame => null;
    public event Action? ViewportMoved { add { } remove { } }
    public event Action? ViewChanged { add { } remove { } }
    public System.Threading.Tasks.Task RestoredViewConfirmed => System.Threading.Tasks.Task.CompletedTask;
    public bool IsComparisonLocked { get; set; }
  }

  private sealed class OneScreen : IScreenInfoProvider
  {
    private static readonly ScreenInfo S = new(new RectI(0, 0, 1920, 1080), 1.5, true);
    public IReadOnlyList<ScreenInfo> Screens => [S];
    public ScreenInfo MainWindowScreen => S;
  }

  /// <summary>
  /// The Liverpool 67P stack: a 1200 × 1200 image whose nucleus is at image px (593.1, 604.3)
  /// (FITS (594.1, 595.7), rows bottom-up), shown at half scale (600 × 600 desktop px: 0.5 desktop
  /// px per image px). "Center on comet" moves the billboard so that pixel lands on the comet:
  /// offset from the image centre (599.5, 599.5) = (−6.4, +4.8) image px = (−3.2, +2.4) desktop px,
  /// turned clockwise on screen by θ: (−3.2 cos θ − 2.4 sin θ, −3.2 sin θ + 2.4 cos θ), and the
  /// centre is the comet minus that. Size and rotation stay; the display's scaling (1.5 here) does
  /// not enter (everything is physical px).
  /// </summary>
  [Theory]
  [InlineData(0.0, 703.2, 297.6)]
  [InlineData(30.0, 703.97128129211, 299.52153903092)]
  public void NucleusPixel_MapsToCometScreenPosition(double rotationDeg, double centreX, double centreY)
  {
    var fake = new FakeAssetRuntime();
    fake.NativeAssets.Add(new ImportedAsset(7, AssetKind.Texture, "lt_67p", "/data/lt_67p.fits", 1200, 1200, [], null, null)
    { IsDisplayable = true });
    var lib = fake.CreateService();
    lib.Refresh();
    var svc = new DesktopBillboardService();
    var vm = new DesktopBillboardsViewModel(svc, lib, new OneScreen(), fake.Schedulers);
    fake.Drain();
    vm.CreateCommand.Execute(null);
    fake.Drain();
    var id = svc.Current.Single().Id;
    svc.Update(id, b => b with
    {
      WidthPx = 600, HeightPx = 600, RotationDeg = rotationDeg, ImageNucleusPx = new PointD(593.1, 604.3),
    });
    var comet = new PointD(700, 300);
    vm.AttachViewport(new CometProbe(comet), poll: false);

    Assert.True(vm.CenterOnComet(id));
    var b = svc.Find(id)!;
    Near(new PointD(centreX, centreY), DesktopBillboardGeometry.Center(b), 1e-9);
    Near(600, b.WidthPx);
    Near(rotationDeg, b.RotationDeg);
    Near(comet, DesktopBillboardGeometry.NucleusDesktop(b), 1e-9);

    // independently, on the drawn quad: the pixel's centre is (u + 0.5) / W along the top edge and
    // (v + 0.5) / H down the left edge from the top-left corner
    var c = DesktopBillboardGeometry.Corners(b);
    double fx = (593.1 + 0.5) / 1200, fy = (604.3 + 0.5) / 1200;
    Near(comet, c[0] + (c[1] - c[0]) * fx + (c[3] - c[0]) * fy, 1e-9);
  }

  /// Rotation is clockwise on screen (y down): a nucleus right of the image centre, turned by
  /// +90°, is below it, so the billboard's centre goes above the comet.
  [Fact]
  public void NucleusOffset_TurnsClockwiseOnScreen()
  {
    var b = new DesktopBillboard(Guid.NewGuid(), "k", 0, 0, 100, 100, RotationDeg: 90)
    {
      ImageWidthPx = 200, ImageHeightPx = 200, ImageNucleusPx = new PointD(99.5 + 20, 99.5),
    };
    Near(new PointD(10, 0), DesktopBillboardGeometry.NucleusLocal(b));
    Near(new PointD(0, 10), DesktopBillboardGeometry.NucleusDesktop(b), 1e-12);
    Near(new PointD(500, 290), DesktopBillboardGeometry.CenterForNucleusAt(b, new PointD(500, 300)), 1e-12);
  }

  /// Without a nucleus pixel (or a native size) the image centre goes on the comet, as before.
  [Fact]
  public void NoNucleusPixel_CentresTheImage()
  {
    var b = new DesktopBillboard(Guid.NewGuid(), "k", 0, 0, 300, 200, RotationDeg: 30) { ImageWidthPx = 600, ImageHeightPx = 400 };
    Near(new PointD(42, 17), DesktopBillboardGeometry.CenterForNucleusAt(b, new PointD(42, 17)));
    var unknownSize = new DesktopBillboard(Guid.NewGuid(), "k", 0, 0, 300, 200) { ImageNucleusPx = new PointD(1, 1) };
    Near(new PointD(42, 17), DesktopBillboardGeometry.CenterForNucleusAt(unknownSize, new PointD(42, 17)));
    // the image centre in pixel-centre coordinates maps to the billboard centre exactly
    Near(new PointD(0, 0), DesktopBillboardGeometry.ImagePxToLocal(b, new PointD(299.5, 199.5))!.Value);
    // the top-left pixel's outer corner (−0.5, −0.5) is the billboard's top-left corner
    Near(new PointD(-150, -100), DesktopBillboardGeometry.ImagePxToLocal(b, new PointD(-0.5, -0.5))!.Value);
  }
}
