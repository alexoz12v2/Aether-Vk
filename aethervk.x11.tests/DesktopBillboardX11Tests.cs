using System.Collections.Immutable;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using AetherVk.Services;
using AetherVk.UI;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Moq;
using Xunit;
using static AetherVk.X11Tests.X11Host;

namespace AetherVk.X11Tests;

/// <summary>
/// Desktop billboards on the real Avalonia X11 backend (plan B-10 … B-15): window type and
/// override_redirect, geometry, displays, left drag, click-through margin, hide on app
/// deactivation, composited alpha. Driven with <c>xdotool</c>, inspected with <c>xwininfo</c>,
/// <c>xprop</c> and <c>import</c>.
/// </summary>
[Collection("X11")]
public sealed class DesktopBillboardX11Tests : IDisposable
{
  private const string TextureKey = "/x11/red.png";

  /// <summary>The main window doubles as an opaque white backdrop at (0, 0) 1600×900.</summary>
  private sealed class Rig : IDisposable
  {
    public Window Main = null!;
    public DesktopBillboardService Service = null!;
    public DesktopBillboardWindowManager Manager = null!;
    public int MainPresses;

    public Rig(bool forceXWayland)
    {
      Environment.SetEnvironmentVariable("AETHERVK_FORCE_XWAYLAND_PATH", forceXWayland ? "1" : null);
      var runtime = new Mock<INativeRuntimeService>();
      var red = Enumerable.Range(0, 8 * 4).SelectMany(_ => new byte[] { 255, 0, 0, 255 }).ToArray();
      runtime.Setup(r => r.GetAssets()).Returns(
      [
        new ImportedAsset(1, AssetKind.Texture, "red", TextureKey, 8, 4, [], null, null) { IsDisplayable = true },
      ]);
      runtime.Setup(r => r.CopyAssetTexels(1)).Returns(new AssetThumbnail(8, 4, red));
      runtime
        .Setup(r => r.RegisterExternalStateListener(It.IsAny<ExternalStateType>(), It.IsAny<Action<nint>>()))
        .Returns(Mock.Of<IDisposable>());
      var storage = new Mock<ILocalStorageService>();
      storage.Setup(s => s.GetSessionPath(It.IsAny<string>())).Returns("/tmp/x11_tests_cache");
      var schedulers = new AvaloniaSchedulerProvider();
      Ui(() =>
      {
        Main = new Window
        {
          Title = "x11-test-main",
          Background = Brushes.White,
          SystemDecorations = SystemDecorations.None,
          Width = 1600,
          Height = 900,
          Position = new PixelPoint(0, 0),
        };
        Main.PointerPressed += (_, _) => MainPresses++;
        Main.Show();
        var lib = new AssetLibraryService(runtime.Object, schedulers, storage.Object);
        lib.Refresh();
        Service = new DesktopBillboardService();
        Manager = new DesktopBillboardWindowManager(
          Service, lib, runtime.Object, new PlatformWindowService(), new MainScreens(Main), schedulers);
        Manager.Attach(Main);
      });
      XDo($"windowfocus --sync {Hex(MainXid)}");
      Settle(300);
    }

    public nint MainXid => Ui(() => Main.TryGetPlatformHandle()!.Handle);

    public DesktopBillboard Add(double cx, double cy, double w = 200, double h = 100, double alpha = 0.5)
    {
      var b = new DesktopBillboard(Guid.NewGuid(), TextureKey, cx, cy, w, h, alpha);
      Ui(() => Service.Restore(Service.Current.Add(b)));
      Assert.True(WaitFor(() => Xid(b.Id) != 0), "billboard window created");
      Settle(300);
      return b;
    }

    public DesktopBillboard Current(Guid id) => Ui(() => Service.Find(id)!);

    public nint Xid(Guid id) =>
      Ui(() => Manager.Windows.FirstOrDefault(w => w.BillboardId == id)?.TryGetPlatformHandle()?.Handle ?? 0);

    public void Dispose()
    {
      Ui(() =>
      {
        Manager.Dispose();
        Main.Close();
      });
      Settle(200);
      Environment.SetEnvironmentVariable("AETHERVK_FORCE_XWAYLAND_PATH", null);
    }
  }

  private sealed class MainScreens(Window main) : IScreenInfoProvider
  {
    public IReadOnlyList<ScreenInfo> Screens =>
      main.Screens.All.Select(AvaloniaScreenInfoProvider.ToInfo).ToList();

    public ScreenInfo MainWindowScreen => Screens[0];
  }

  public DesktopBillboardX11Tests()
  {
    Skip.If(SkipReason is not null, SkipReason);
    EnsureStarted();
  }

  public void Dispose()
  {
  }

  /// B-10: plain X11 billboards are UTILITY windows; forced XWayland ones are override_redirect;
  /// either way the window sits where the geometry says.
  [SkippableTheory]
  [InlineData(false)]
  [InlineData(true)]
  public void WindowTypeAndGeometry(bool forceXWayland)
  {
    using var rig = new Rig(forceXWayland);
    var b = rig.Add(700, 400);
    var xid = rig.Xid(b.Id);
    var info = XWinInfo(xid);
    var want = DesktopBillboardGeometry.WindowRect(b);
    Assert.Equal((want.X, want.Y, want.Width, want.Height), (info.X, info.Y, info.Width, info.Height));
    Assert.True(info.Viewable);
    Assert.Equal(forceXWayland, info.OverrideRedirect);
    if (!forceXWayland)
      Assert.Contains("_NET_WM_WINDOW_TYPE_UTILITY", XProp(xid, "_NET_WM_WINDOW_TYPE"));
  }

  /// B-11: two monitors side by side (xrandr --setmonitor on the Xvfb screen); a billboard on the
  /// right one is on display 2 and its window there.
  [SkippableFact]
  public void SecondMonitor()
  {
    using var rig = new Rig(false);
    var screens = Ui(() => new MainScreens(rig.Main).Screens);
    // x11_e2e.sh sets up two RandR monitors and says so: then fewer is a failure, not a skip
    if (Environment.GetEnvironmentVariable("X11_E2E_MONITORS") == "2")
      Assert.Equal(2, screens.Count);
    Skip.If(screens.Count < 2, $"the X server reports {screens.Count} monitor(s); x11_e2e.sh sets up two");
    var right = screens.OrderBy(s => s.Bounds.X).Last();
    var b = rig.Add(right.Bounds.X + 300, 300);
    Assert.Equal(screens.ToList().IndexOf(right),
      DesktopBillboardGeometry.DisplayOf(screens, new PointD(b.CenterX, b.CenterY)));
    var info = XWinInfo(rig.Xid(b.Id));
    Assert.True(right.Bounds.Contains(info.X + info.Width / 2.0, info.Y + info.Height / 2.0));
  }

  /// B-12: a real left drag (XTest) moves the billboard by the drag and the window with it.
  [SkippableTheory]
  [InlineData(false)]
  [InlineData(true)]
  public void LeftDragMovesTheWindow(bool forceXWayland)
  {
    using var rig = new Rig(forceXWayland);
    var b = rig.Add(700, 400);
    int x0 = 680, y0 = 410;
    XDo($"mousemove {x0} {y0} sleep 0.15 mousedown 1 sleep 0.15 mousemove {x0 + 20} {y0 + 10} sleep 0.15 "
      + $"mousemove {x0 + 60} {y0 + 30} sleep 0.15 mouseup 1");
    Assert.True(WaitFor(() => Math.Abs(rig.Current(b.Id).CenterX - 760) < 0.5), $"centre {rig.Current(b.Id).CenterX}");
    var now = rig.Current(b.Id);
    Assert.Equal(430, now.CenterY, 1);
    Settle(300);
    var info = XWinInfo(rig.Xid(b.Id));
    var want = DesktopBillboardGeometry.WindowRect(now);
    Assert.Equal((want.X, want.Y), (info.X, info.Y));
  }

  /// B-13: the transparent margin lets clicks through to the window behind; the image does not.
  [SkippableFact]
  public void MarginClicksReachTheWindowBehind()
  {
    using var rig = new Rig(false);
    var b = rig.Add(700, 400);
    var win = DesktopBillboardGeometry.WindowRect(b);
    int before = Ui(() => rig.MainPresses);
    XDo($"mousemove {win.X + 5} {win.Y + 5} sleep 0.1 click 1");
    Assert.True(WaitFor(() => Ui(() => rig.MainPresses) == before + 1), "margin click reached the main window");
    XDo("mousemove 650 410 sleep 0.1 click 1");
    Settle(300);
    Assert.Equal(before + 1, Ui(() => rig.MainPresses));
    // the gizmo handles take clicks once shown: right click cycles to Translate
    XDo("mousemove 650 410 sleep 0.1 click 3");
    Assert.True(WaitFor(() => rig.Current(b.Id).GizmoMode == BillboardGizmoMode.Translate));
    var arrow = DesktopBillboardGeometry.Handles(rig.Current(b.Id), BillboardGizmoMode.Translate)
      .Single(h => h.Kind == BillboardHandle.MoveAlongX).Center;
    Settle(300);
    XDo($"mousemove {(int)arrow.X} {(int)arrow.Y} sleep 0.1 click 1");
    Settle(300);
    Assert.Equal(before + 1, Ui(() => rig.MainPresses));
  }

  /// B-14: XWayland path: billboards hide when focus leaves the application and come back with it;
  /// focus moving to a billboard does not hide them.
  [SkippableFact]
  public void HiddenWhileTheAppIsInTheBackground()
  {
    using var rig = new Rig(true);
    var b = rig.Add(700, 400);
    var xid = rig.Xid(b.Id);
    Assert.True(XWinInfo(xid).Viewable);
    var other = Ui(() =>
    {
      var w = new Window { Title = "x11-test-other", Width = 200, Height = 100, Position = new PixelPoint(1300, 700) };
      w.Show();
      return w;
    });
    Settle(300);
    var otherXid = Ui(() => other.TryGetPlatformHandle()!.Handle);
    XDo($"windowfocus --sync {Hex(otherXid)}");
    Assert.True(WaitFor(() => !XWinInfo(xid).Viewable), "hidden while another window has the focus");
    XDo($"windowfocus --sync {Hex(rig.MainXid)}");
    Assert.True(WaitFor(() => XWinInfo(xid).Viewable), "shown again when the app is back");
    // clicking the billboard keeps it shown
    XDo("mousemove 650 410 sleep 0.1 click 1");
    Settle(600);
    Assert.True(XWinInfo(xid).Viewable);
    Ui(() => other.Close());
  }

  /// B-15: composited: the margin shows the white window behind, the image is red at 50 %.
  [SkippableFact]
  public void CompositedAlpha()
  {
    Skip.IfNot(CompositorRunning(), "no compositing manager (install xcompmgr; x11_e2e.sh starts it)");
    using var rig = new Rig(false);
    var b = rig.Add(700, 400, alpha: 0.5);
    var win = DesktopBillboardGeometry.WindowRect(b);
    Settle(500);
    var margin = RootPixel(win.X + 4, win.Y + 4);
    var image = RootPixel(680, 410);
    Assert.NotNull(margin);
    Assert.NotNull(image);
    Assert.Equal((255, 255, 255), margin!.Value);
    // 0.5 red over white = (255, 128, 128)
    Assert.InRange(image!.Value.R, 250, 255);
    Assert.InRange(image.Value.G, 118, 138);
    Assert.InRange(image.Value.B, 118, 138);
  }
}
