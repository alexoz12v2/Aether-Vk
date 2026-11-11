using System.Diagnostics;
using Avalonia;
using Avalonia.Automation;
using Avalonia.Controls;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.VisualTree;
using AetherVk.Controls;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using Xunit;
using Xunit.Abstractions;
using static AetherVk.X11Tests.X11Host;

namespace AetherVk.X11Tests;

/// <summary>
/// A second live Vulkan view next to the main viewport, on the real application: a
/// <see cref="VulkanViewportControl"/> with the <see cref="ViewportRole.NucleusMini"/> role inside an
/// <see cref="Expander"/> in a panel added to the main window outside the docking tree. Its surface
/// shares the main one's X connection (<see cref="X11SurfaceDisplay"/>). The tests check what the native host does with it: when its window and engine are created, that expanding,
/// collapsing and hiding keep both, that a hidden second view does not stall the main one, and that
/// the application shuts down cleanly with both attached.
/// </summary>
[Collection("X11")]
public sealed class SecondViewportX11Tests
{
  private const string HostId = "SecondViewportHost";
  private const string ExpanderId = "SecondViewportExpander";
  private const string PanelId = "SecondViewportPanel";

  private readonly ITestOutputHelper _output;

  public SecondViewportX11Tests(ITestOutputHelper output)
  {
    _output = output;
    Skip.If(SkipReason is not null, SkipReason);
    EnsureStarted();
  }

  private void Report(string s)
  {
    _output.WriteLine(s);
    Console.Error.WriteLine($"[SecondViewportX11Tests] {s}");
  }

  /// <summary>The rig with the second view's panel in its main window (the Expander collapsed).</summary>
  private sealed class Scene
  {
    public required AppRig Rig { get; init; }
    public required NativeRuntimeService Runtime { get; init; }
    public required ulong MainPe { get; init; }
    public required ulong MainCamera { get; init; }
    public required VulkanViewportControlViewModel Second { get; init; }
    public required Border Panel { get; init; }
    public required Expander Expander { get; init; }

    public ulong SecondPe => Ui(() => Second.PresentationEngineId);
    public nint SecondXid => Ui(() => Second.NativeHandle);
    public int SecondInitializations => Ui(() => Second.InitializeCount);

    public void SetExpanded(bool expanded, int settleMs = 200)
    {
      Ui(() => Expander.IsExpanded = expanded);
      Settle(settleMs);
    }

    public void SetPanelVisible(bool visible, int settleMs = 200)
    {
      Ui(() => Panel.IsVisible = visible);
      Settle(settleMs);
    }
  }

  private static Scene Build(AppRig rig)
  {
    var runtime = Assert.IsType<NativeRuntimeService>(rig.Host.Services.GetRequiredService<INativeRuntimeService>());
    Assert.True(WaitFor(() => runtime.PresentationEngineId is not null, 30_000), "the main viewport attached its engine");
    ulong mainPe = runtime.PresentationEngineId!.Value;
    ulong mainCamera = runtime.CameraEntityId ?? 0;
    Assert.Null(runtime.NucleusMiniPresentationEngineId);
    // this second view has no owner to resume it: it presents from its creation, as before the
    // application's mini-view started paused
    runtime.NucleusMiniPresentFlags = ViewportPresentFlags.LowLatency;

    var services = rig.Host.Services;
    var (second, panel, expander) = Ui(() =>
    {
      var mainViewport = rig.Main.GetVisualDescendants().Select(v => v.DataContext).OfType<Viewport3DViewModel>().First();
      // as the application's factory builds the main one, in the mini-view role: an engine and nothing else
      var vm = new VulkanViewportControlViewModel(
        services.GetRequiredService<IWindowInputRouter>(),
        services.GetRequiredService<INativeInputHandlerFactory>(),
        runtime,
        mainViewport)
      {
        Role = ViewportRole.NucleusMini,
      };
      var host = new VulkanViewportControl { DataContext = vm, Width = 320, Height = 240 };
      AutomationProperties.SetAutomationId(host, HostId);
      var exp = new Expander { Header = "Second view", IsExpanded = false, Content = host };
      AutomationProperties.SetAutomationId(exp, ExpanderId);
      var border = new Border
      {
        Child = exp,
        Width = 360,
        Background = Brushes.Black,
        HorizontalAlignment = HorizontalAlignment.Left,
        VerticalAlignment = VerticalAlignment.Top,
      };
      AutomationProperties.SetAutomationId(border, PanelId);
      // over the docking tree's row, not inside it: nothing of the layout is re-parented
      var root = (Grid)rig.Main.Content!;
      Grid.SetRow(border, root.RowDefinitions.Count - 1);
      root.Children.Add(border);
      return (vm, border, exp);
    });
    Settle(1000);
    return new Scene
    {
      Rig = rig,
      Runtime = runtime,
      MainPe = mainPe,
      MainCamera = mainCamera,
      Second = second,
      Panel = panel,
      Expander = expander,
    };
  }

  /// <summary>The X server still has the window.</summary>
  private static bool XidAlive(nint xid) =>
    xid != 0 && Run("xwininfo", $"-id {Hex(xid)}").Contains("Map State:", StringComparison.Ordinal);

  private static bool XidViewable(nint xid) =>
    xid != 0 && Run("xwininfo", $"-id {Hex(xid)}").Contains("Map State: IsViewable", StringComparison.Ordinal);

  /// <summary>Expands and waits for the second view's engine and mapped window.</summary>
  private static void ExpandAndRequire(Scene s)
  {
    s.SetExpanded(true);
    Assert.True(WaitFor(() => s.SecondPe != 0, 15_000), "the second view attached a presentation engine once expanded");
    Assert.True(WaitFor(() => XidViewable(s.SecondXid), 10_000), "the second view's window is mapped once expanded");
  }

  /// <summary>The main viewport's cached ids are the main viewport's, the second engine is the mini-view's.</summary>
  private static void AssertMainUntouched(Scene s)
  {
    Assert.Equal(s.MainPe, s.Runtime.PresentationEngineId);
    Assert.Equal(s.MainCamera, s.Runtime.CameraEntityId ?? 0);
    Assert.NotEqual(s.MainPe, s.SecondPe);
    Assert.Equal(s.SecondPe, s.Runtime.NucleusMiniPresentationEngineId);
    var mainViewport = Ui(() => s.Rig.Main.GetVisualDescendants().Select(v => v.DataContext).OfType<Viewport3DViewModel>().First());
    Assert.Equal(s.MainPe, Ui(() => mainViewport.PresentationEngineId));
    Assert.Equal(s.MainCamera, Ui(() => mainViewport.CameraId));
  }

  [SkippableFact]
  public void SecondViewport_CreatesItsNativeWindowWhileCollapsed_OrOnFirstExpand()
  {
    using var rig = new AppRig();
    var s = Build(rig);

    Report($"collapsed: the host is {(rig.Find<VulkanViewportControl>(HostId) is null ? "NOT in" : "in")} the visual tree");
    nint xidBefore = s.SecondXid;
    ulong peBefore = s.SecondPe;
    bool viewableBefore = XidViewable(xidBefore);
    Report($"collapsed, before the first expand: XID {Hex(xidBefore)}, PE {peBefore}, mapped {viewableBefore}, "
      + $"initializations {s.SecondInitializations}");

    ExpandAndRequire(s);
    nint xidAfter = s.SecondXid;
    ulong peAfter = s.SecondPe;
    Report($"after the first expand: XID {Hex(xidAfter)}, PE {peAfter}, mapped {XidViewable(xidAfter)}, "
      + $"initializations {s.SecondInitializations}");
    Report(peBefore != 0
      ? "CASE: the native window and its engine are created WHILE COLLAPSED (before the first expand)"
      : "CASE: the native window and its engine are created ON THE FIRST EXPAND");

    Assert.NotEqual(0UL, peAfter);
    Assert.NotEqual((nint)0, xidAfter);
    Assert.True(XidAlive(xidAfter), "the X server has the second view's window");
    if (peBefore != 0)
    {
      // created while collapsed: the expand must show that same window and engine
      Assert.Equal(xidBefore, xidAfter);
      Assert.Equal(peBefore, peAfter);
    }
    Assert.Equal(1, s.SecondInitializations);
    AssertMainUntouched(s);
  }

  [SkippableFact]
  public void SecondViewport_ExpandCollapse20x_KeepsTheSameNativeWindowAndPe()
  {
    using var rig = new AppRig();
    var s = Build(rig);
    ExpandAndRequire(s);
    nint xid = s.SecondXid;
    ulong pe = s.SecondPe;
    Report($"expanded: XID {Hex(xid)}, PE {pe}");

    void Same(string when)
    {
      Assert.True(s.SecondXid == xid, $"{when}: the native window changed, {Hex(xid)} → {Hex(s.SecondXid)}");
      Assert.True(s.SecondPe == pe, $"{when}: the presentation engine changed, {pe} → {s.SecondPe}");
      Assert.True(s.SecondInitializations == 1, $"{when}: the native window was handed over {s.SecondInitializations} times");
      Assert.True(XidAlive(xid), $"{when}: the X server lost window {Hex(xid)}");
      Assert.Equal(pe, s.Runtime.NucleusMiniPresentationEngineId);
    }

    int unmappedWhenCollapsed = 0, mappedWhenExpanded = 0;
    for (int i = 1; i <= 20; i++)
    {
      s.SetExpanded(false);
      Same($"collapse {i}");
      if (!XidViewable(xid)) unmappedWhenCollapsed++;
      s.SetExpanded(true);
      Same($"expand {i}");
      if (WaitFor(() => XidViewable(xid), 2000)) mappedWhenExpanded++;
    }
    Report($"20 expand/collapse cycles: unmapped when collapsed {unmappedWhenCollapsed}/20, mapped when expanded {mappedWhenExpanded}/20");
    Assert.Equal(20, mappedWhenExpanded);

    int unmappedWhenHidden = 0, mappedWhenShown = 0;
    for (int i = 1; i <= 10; i++)
    {
      s.SetPanelVisible(false);
      Same($"panel hidden {i}");
      if (!XidViewable(xid)) unmappedWhenHidden++;
      s.SetPanelVisible(true);
      Same($"panel shown {i}");
      if (WaitFor(() => XidViewable(xid), 2000)) mappedWhenShown++;
    }
    Report($"10 panel hide/show cycles: unmapped when hidden {unmappedWhenHidden}/10, mapped when shown {mappedWhenShown}/10");
    Assert.Equal(10, mappedWhenShown);

    AssertMainUntouched(s);
  }

  [SkippableFact]
  public void SecondViewport_HiddenDoesNotStallTheMainViewport()
  {
    using var rig = new AppRig();
    var s = Build(rig);
    // both engines exist in both phases (the engine may only be created on the first expand)
    ExpandAndRequire(s);
    ulong secondPe = s.SecondPe;
    nint xid = s.SecondXid;
    AssertMainUntouched(s);

    // A capture is taken by a frame of that engine submitted after the request, or it fails: each
    // one that returns an image is a frame the render thread rendered for the main viewport now.
    long[] Sample(string phase)
    {
      // the first capture of a phase is not timed (one-off allocations)
      Assert.True(s.Runtime.CaptureViewportAsync(s.MainPe, hideLabels: false).GetAwaiter().GetResult() is not null,
        $"{phase}: the main viewport rendered no frame at all");
      var ms = new long[5];
      for (int i = 0; i < ms.Length; i++)
      {
        var sw = Stopwatch.StartNew();
        var image = s.Runtime.CaptureViewportAsync(s.MainPe, hideLabels: false).GetAwaiter().GetResult();
        ms[i] = sw.ElapsedMilliseconds;
        Assert.True(image is not null, $"{phase}, sample {i + 1}: the main viewport rendered no frame ({ms[i]} ms)");
        Assert.True(image!.Width > 0 && image.Height > 0, $"{phase}, sample {i + 1}: empty frame");
        Thread.Sleep(200);
      }
      Report($"{phase}: main viewport frame captured in [{string.Join(", ", ms)}] ms");
      return ms;
    }

    // what the second engine itself does in a phase: reported, not required
    void ReportSecond(string phase)
    {
      var sw = Stopwatch.StartNew();
      var image = s.Runtime.CaptureViewportAsync(secondPe, hideLabels: false).GetAwaiter().GetResult();
      Report(image is null
        ? $"{phase}: the second view presented no frame within {sw.ElapsedMilliseconds} ms"
        : $"{phase}: the second view presented a {image.Width}×{image.Height} frame in {sw.ElapsedMilliseconds} ms");
    }

    s.SetExpanded(false, settleMs: 500);
    bool unmapped = WaitFor(() => !XidViewable(xid), 3000);
    Report($"collapsed: the second view's window {Hex(xid)} is {(unmapped ? "unmapped" : "STILL MAPPED")}");
    var hidden = Sample("second view collapsed");
    ReportSecond("second view collapsed");

    ExpandAndRequire(s);
    Settle(500);
    var shown = Sample("second view expanded");
    ReportSecond("second view expanded");

    s.SetPanelVisible(false, settleMs: 500);
    Report($"panel hidden: the second view's window is {(XidViewable(xid) ? "STILL MAPPED" : "unmapped")}");
    var panelHidden = Sample("second view's panel hidden");
    s.SetPanelVisible(true, settleMs: 500);

    Assert.All(hidden, t => Assert.True(t < 1000, $"second view collapsed: a main frame took {t} ms"));
    Assert.All(panelHidden, t => Assert.True(t < 1000, $"second view's panel hidden: a main frame took {t} ms"));
    Assert.All(shown, t => Assert.True(t < 1000, $"second view expanded: a main frame took {t} ms"));
    Assert.Equal(secondPe, s.SecondPe);
    AssertMainUntouched(s);
  }

  [SkippableFact]
  public void SecondViewport_ShutdownWithTwoViewports_IsClean()
  {
    for (int round = 1; round <= 5; round++)
    {
      var rig = new AppRig();
      bool disposed = false;
      try
      {
        var s = Build(rig);
        ExpandAndRequire(s);
        ulong secondPe = s.SecondPe;
        AssertMainUntouched(s);
        // even rounds shut down with the second view collapsed (its window unmapped)
        if (round % 2 == 0)
          s.SetExpanded(false, settleMs: 500);
        Report($"round {round}: main PE {s.MainPe}, second PE {secondPe}, second view "
          + $"{(round % 2 == 0 ? "collapsed" : "expanded")}, mapped {XidViewable(s.SecondXid)}");

        // read while the engine is still up: both engines must be detached by then
        bool hookRan = false;
        ulong? mainAtEngineStop = ulong.MaxValue;
        ulong? secondaryAtEngineStop = ulong.MaxValue;
        rig.BeforeEngineShutdown = () =>
        {
          mainAtEngineStop = s.Runtime.PresentationEngineId;
          secondaryAtEngineStop = s.Runtime.NucleusMiniPresentationEngineId;
          hookRan = true;
        };

        var sw = Stopwatch.StartNew();
        disposed = true;
        rig.Dispose();
        Report($"round {round}: shutdown took {sw.ElapsedMilliseconds} ms, completed {rig.ShutdownCompleted}, "
          + $"error {rig.ShutdownError?.GetType().Name ?? "none"}");

        Assert.True(rig.ShutdownError is null, $"round {round}: the shutdown threw: {rig.ShutdownError}");
        Assert.True(rig.ShutdownCompleted, $"round {round}: the shutdown did not finish within its time limit");
        Assert.True(hookRan, $"round {round}: the shutdown never reached the engine stop");
        Assert.True(mainAtEngineStop is null, $"round {round}: the main engine {mainAtEngineStop} was still attached at the engine stop");
        Assert.True(secondaryAtEngineStop is null,
          $"round {round}: the second engine {secondaryAtEngineStop} was still attached at the engine stop");
        Assert.Null(s.Runtime.PresentationEngineId);
        Assert.NotEqual(secondPe, s.Runtime.NucleusMiniPresentationEngineId);
        Assert.Null(s.Runtime.NucleusMiniPresentationEngineId);
      }
      finally
      {
        if (!disposed)
          rig.Dispose();
      }
    }
  }
}
