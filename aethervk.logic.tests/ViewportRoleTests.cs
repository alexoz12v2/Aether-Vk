using System;
using System.Linq;
using System.Reactive.Disposables;
using Microsoft.Reactive.Testing;
using AetherVk.Logic.Input;
using AetherVk.Logic.Services;
using AetherVk.Logic.Services.NativeInput;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The viewport roles: one main viewport, one nucleus mini-view beside it, the mini removed first at
/// shutdown, and the mini's camera never registered (so the mode bar, settings, billboards and the
/// save host, which only see registered cameras, never see it).
/// </summary>
[Collection("Sequential")]
public class ViewportRoleTests
{
  // ── Role table ───────────────────────────────────────────────────────────────

  [Fact]
  public void AddViewport_AllowsMainPlusOneMini_RefusesThird()
  {
    var table = new ViewportRoleTable();

    Assert.True(table.CanAdd(ViewportRole.Main, out _));
    Assert.True(table.TryAdd(ViewportRole.Main, 10, 11));
    Assert.True(table.CanAdd(ViewportRole.NucleusMini, out _));
    Assert.True(table.TryAdd(ViewportRole.NucleusMini, 20, 21));

    // a third viewport, of either role, is refused with a reason and records nothing
    Assert.False(table.CanAdd(ViewportRole.Main, out var mainRefusal));
    Assert.False(string.IsNullOrEmpty(mainRefusal));
    Assert.False(table.CanAdd(ViewportRole.NucleusMini, out var miniRefusal));
    Assert.False(string.IsNullOrEmpty(miniRefusal));
    Assert.False(table.TryAdd(ViewportRole.Main, 30, 31));
    Assert.False(table.TryAdd(ViewportRole.NucleusMini, 40, 41));
    Assert.False(table.CanAdd((ViewportRole)99, out _));

    Assert.Equal(10UL, table.MainPresentationEngineId);
    Assert.Equal(11UL, table.MainCameraEntityId);
    Assert.Equal(20UL, table.NucleusMiniPresentationEngineId);
    Assert.Equal(21UL, table.NucleusMiniCameraEntityId);
  }

  [Fact]
  public void Mini_Refused_WithoutMain()
  {
    var table = new ViewportRoleTable();
    Assert.False(table.CanAdd(ViewportRole.NucleusMini, out var refusal));
    Assert.False(string.IsNullOrEmpty(refusal));
    Assert.False(table.TryAdd(ViewportRole.NucleusMini, 20, 21));
    Assert.Null(table.NucleusMiniPresentationEngineId);
    Assert.Null(table.MainPresentationEngineId);

    // and again once the main viewport went away
    Assert.True(table.TryAdd(ViewportRole.Main, 10, 11));
    Assert.Equal(ViewportRole.Main, table.Removed(10));
    Assert.False(table.CanAdd(ViewportRole.NucleusMini, out _));
  }

  [Fact]
  public void RemoveMini_KeepsMainIds()
  {
    var table = new ViewportRoleTable();
    Assert.True(table.TryAdd(ViewportRole.Main, 10, 11));
    Assert.True(table.TryAdd(ViewportRole.NucleusMini, 20, 21));

    Assert.Equal(ViewportRole.NucleusMini, table.Removed(20));
    Assert.Null(table.NucleusMiniPresentationEngineId);
    Assert.Null(table.NucleusMiniCameraEntityId);
    Assert.Equal(10UL, table.MainPresentationEngineId);
    Assert.Equal(11UL, table.MainCameraEntityId);

    // unknown and zero engines change nothing; the role is free again
    Assert.Null(table.Removed(20));
    Assert.Null(table.Removed(0));
    Assert.Equal(10UL, table.MainPresentationEngineId);
    Assert.True(table.TryAdd(ViewportRole.NucleusMini, 22, 23));
    Assert.Equal(22UL, table.NucleusMiniPresentationEngineId);
  }

  [Fact]
  public void Shutdown_Order_MiniFirst()
  {
    var table = new ViewportRoleTable();
    Assert.Empty(table.ShutdownOrder());
    Assert.True(table.TryAdd(ViewportRole.Main, 10, 11));
    Assert.Equal(new ulong[] { 10 }, table.ShutdownOrder());
    Assert.True(table.TryAdd(ViewportRole.NucleusMini, 20, 21));
    Assert.Equal(new ulong[] { 20, 10 }, table.ShutdownOrder());

    // the shutdown path the application and the X11 rig use
    var runtime = new Mock<INativeRuntimeService>();
    runtime.Setup(r => r.PresentationEngineId).Returns(10UL);
    runtime.Setup(r => r.NucleusMiniPresentationEngineId).Returns(20UL);
    runtime.Object.RemoveAllViewports();
    var removed = runtime.Invocations
      .Where(i => i.Method.Name == nameof(INativeRuntimeService.RemoveViewport))
      .Select(i => (ulong)i.Arguments[0])
      .ToArray();
    Assert.Equal(new ulong[] { 20, 10 }, removed);

    // nothing attached: nothing removed
    var empty = new Mock<INativeRuntimeService>();
    empty.Object.RemoveAllViewports();
    empty.Verify(r => r.RemoveViewport(It.IsAny<ulong>()), Times.Never);
  }

  // ── Camera registration ──────────────────────────────────────────────────────

  private sealed class FakeHandler : INativeInputHandlerSubscribable
  {
    public int KeystrokeSubscriptions;
    public int MouseSubscriptions;
    public void FocusViewportWindow() { }
    public void Dispose() { }

    public IDisposable SubscribeComposedKeystrokes(Action<ComposedKeystroke> onNext)
    {
      KeystrokeSubscriptions++;
      return Disposable.Empty;
    }

    public IDisposable SubscribeNativeMouseEvents(Action<NativeMouseEvent> onNext)
    {
      MouseSubscriptions++;
      return Disposable.Empty;
    }
  }

  private sealed record Rig(
    Mock<INativeRuntimeService> Runtime,
    Mock<ICameraServiceRegistry> Registry,
    Viewport3DViewModel Viewport,
    VulkanViewportControlViewModel View,
    FakeHandler Handler);

  /// <summary>The main viewport's view model with a mocked registry, and a native view of
  /// <paramref name="role"/> beside it, as the application's factory builds one.</summary>
  private static Rig Build(ViewportRole role, ulong pe, ulong cam)
  {
    var dispatcher = new Mock<IUiThreadDispatcher>();
    dispatcher.Setup(d => d.Dispatch(It.IsAny<Action>())).Callback<Action>(a => a());
    dispatcher.Setup(d => d.CheckAccess()).Returns(true);

    var sp = new Mock<ISchedulerProvider>();
    sp.Setup(s => s.MainThread).Returns(System.Reactive.Concurrency.ImmediateScheduler.Instance);
    sp.Setup(s => s.Background).Returns(new TestScheduler());
    var schedulers = sp.Object;

    var runtime = new Mock<INativeRuntimeService>();
    ulong outPe = pe, outCam = cam;
    runtime.Setup(r => r.AddViewport(
        It.IsAny<uint>(), It.IsAny<uint>(), It.IsAny<string>(), It.IsAny<Func<CNativeWindowHandle>?>(),
        It.IsAny<uint>(), out outPe, out outCam, It.IsAny<ViewportRole>()))
      .Returns(true);

    var registry = new Mock<ICameraServiceRegistry>();
    var breadcrumb = new BreadcrumbService();
    var cometConfig = new CometConfigService(runtime.Object, schedulers);
    var timeline = new TimelineService(runtime.Object, schedulers, cometConfig, breadcrumb);
    var cometTracker = new CometPositionTrackerService(runtime.Object, schedulers, timeline);
    var cameraService = new CameraService(runtime.Object, schedulers, cometTracker, cometConfig, breadcrumb,
      Mock.Of<ICometMessenger>(), registry.Object);

    var sessionService = new Mock<ITabStateService<ViewportSession>>();
    var sessionList = new[] { new SessionId(typeof(ViewportSession), 1) };
    sessionService.Setup(s => s.ActiveSessionIds).Returns(sessionList);
    sessionService.Setup(s => s.ObserveSessionList()).Returns(System.Reactive.Linq.Observable.Return<System.Collections.Generic.IReadOnlyList<SessionId>>(sessionList));
    sessionService.Setup(s => s.ObserveSession(It.IsAny<SessionId>())).Returns(System.Reactive.Linq.Observable.Return(new ViewportSession()));
    sessionService.Setup(s => s.GetSession(It.IsAny<SessionId>())).Returns(new ViewportSession());

    VulkanViewportControlViewModel MainVulkanFactory(Viewport3DViewModel vm) =>
      new Mock<VulkanViewportControlViewModel>(
        MockBehavior.Loose,
        new Mock<IWindowInputRouter>().Object,
        new Mock<INativeInputHandlerFactory>().Object,
        runtime.Object,
        vm
      ).Object;

    ViewportOverlayViewModel OverlayFactory(Viewport3DViewModel vm) =>
      new Mock<ViewportOverlayViewModel>(
        MockBehavior.Loose,
        cameraService,
        runtime.Object,
        new BreadcrumbService(),
        dispatcher.Object,
        new Mock<IFileDialogService>().Object,
        new Mock<ITabStateService<TimelineSession>>().Object,
        cometConfig,
        vm,
        new AetherVk.Logic.Tests.Mocks.TestSchedulerProvider(),
        timeline,
        new Mock<ITabStateService<ModelSession>>().Object,
        (PreferencesService?)null,
        (ITranslationService?)null
      ).Object;

    var viewport = new Viewport3DViewModel(
      runtime.Object,
      breadcrumb,
      dispatcher.Object,
      new Mock<IFileDialogService>().Object,
      cameraService,
      MainVulkanFactory,
      OverlayFactory,
      new Mock<IPlatformWindowService>().Object,
      new Mock<IWindowInputRouter>().Object,
      sessionService.Object
    );

    var handler = new FakeHandler();
    var handlerFactory = new Mock<INativeInputHandlerFactory>();
    handlerFactory
      .Setup(f => f.Create(It.IsAny<IntPtr>(), It.IsAny<string>(), It.IsAny<TraceLevel>()))
      .Returns(handler);
    var view = new VulkanViewportControlViewModel(
      new Mock<IWindowInputRouter>().Object, handlerFactory.Object, runtime.Object, viewport)
    {
      Role = role,
    };
    return new Rig(runtime, registry, viewport, view, handler);
  }

  private static void VerifyAddedWith(Rig rig, ViewportRole role) =>
    rig.Runtime.Verify(r => r.AddViewport(
      It.IsAny<uint>(), It.IsAny<uint>(), It.IsAny<string>(), It.IsAny<Func<CNativeWindowHandle>?>(),
      It.IsAny<uint>(), out It.Ref<ulong>.IsAny, out It.Ref<ulong>.IsAny, role), Times.Once);

  [Fact]
  public void MiniView_NeverRegistersItsCamera_KeysGoToTheMainContext()
  {
    var rig = Build(ViewportRole.NucleusMini, pe: 20, cam: 21);

    Assert.True(rig.View.InitializeHandle(new IntPtr(0x1234), "XID"));

    VerifyAddedWith(rig, ViewportRole.NucleusMini);
    Assert.Equal(20UL, rig.View.PresentationEngineId);
    Assert.Equal(21UL, rig.View.CameraEntityId);
    // the registry is all the mode bar, settings, billboards and the save host see
    rig.Registry.Verify(r => r.RegisterSelf(It.IsAny<ulong>(), It.IsAny<CameraService>()), Times.Never);
    // the main viewport's view model (export target) never takes the mini's ids
    Assert.Equal(0UL, rig.Viewport.PresentationEngineId);
    Assert.Equal(0UL, rig.Viewport.CameraId);
    // keys typed while its window has the focus are forwarded to the "Viewport" context (the main
    // viewport's handler, NucleusMiniViewModelTests); without a drag handler it takes no mouse
    Assert.Equal(1, rig.Handler.KeystrokeSubscriptions);
    Assert.Equal(0, rig.Handler.MouseSubscriptions);

    // its size and screen position stay its own
    uint width = rig.Viewport.Width, height = rig.Viewport.Height;
    rig.View.NotifyResized(123, 45);
    Assert.Equal(width, rig.Viewport.Width);
    Assert.Equal(height, rig.Viewport.Height);
  }

  [Fact]
  public void MainView_RegistersItsCamera()
  {
    // the control for the test above: the same rig in the main role does register
    var rig = Build(ViewportRole.Main, pe: 10, cam: 11);

    Assert.True(rig.View.InitializeHandle(new IntPtr(0x1234), "XID"));

    VerifyAddedWith(rig, ViewportRole.Main);
    rig.Registry.Verify(r => r.RegisterSelf(11UL, It.IsAny<CameraService>()), Times.Once);
    Assert.Equal(10UL, rig.Viewport.PresentationEngineId);
    Assert.Equal(11UL, rig.Viewport.CameraId);
    Assert.Equal(1, rig.Handler.KeystrokeSubscriptions);
    Assert.Equal(1, rig.Handler.MouseSubscriptions);
  }
}
