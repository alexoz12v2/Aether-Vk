using System;
using System.Numerics;
using System.Reactive.Concurrency;
using AetherVk.Logic.Input;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// Unit tests for <see cref="Viewport3DViewModel"/> focusing on the operator-stack input pipeline
/// and camera-mode switching. Native FFI paths are bypassed via a mocked
/// <see cref="INativeRuntimeService"/>.
/// </summary>
[Collection("Sequential")]
public class Viewport3DViewModelTests
{
  // ── Helpers ─────────────────────────────────────────────────────────────────

  private static ISchedulerProvider MakeImmediateSchedulers()
  {
    var sp = new Mock<ISchedulerProvider>();
    sp.Setup(s => s.MainThread).Returns(ImmediateScheduler.Instance);
    sp.Setup(s => s.Background).Returns(ImmediateScheduler.Instance);
    return sp.Object;
  }

  /// <summary>
  /// Builds a fully wired <see cref="Viewport3DViewModel"/> using mocked native services
  /// so no native DLL is required.
  /// </summary>
  /// <param name="cometCommitted">
  /// When <c>true</c>, pre-configures <see cref="CometConfigService"/> as having a committed
  /// almanac so that tests exercising <see cref="CameraMode.CometOrbiting"/> can enter that mode.
  /// </param>
  private static Viewport3DViewModel BuildVm(bool cometCommitted = false)
  {
    var dispatcher = new Mock<IUiThreadDispatcher>();
    dispatcher
      .Setup(d => d.Dispatch(It.IsAny<Action>()))
      .Callback<Action>(a => a());
    dispatcher
      .Setup(d => d.CheckAccess())
      .Returns(true);

    var schedulers    = MakeImmediateSchedulers();
    var runtime       = new Mock<INativeRuntimeService>();
    var breadcrumb    = new BreadcrumbService(dispatcher.Object);
    var timeline      = new TimelineService(runtime.Object, schedulers);
    var cometTracker  = new CometPositionTrackerService(runtime.Object, schedulers, timeline);
    var cometConfig   = new CometConfigService(runtime.Object, schedulers);
    var cameraService = new CameraService(runtime.Object, schedulers, cometTracker, cometConfig, breadcrumb);

    // Pre-set the committed state so tests that switch to CometOrbiting work.
    if (cometCommitted)
    {
      var field = typeof(CometConfigService)
        .GetField("_isCommittedSubject", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
      var subject = (System.Reactive.Subjects.BehaviorSubject<bool>?)field?.GetValue(cometConfig);
      subject?.OnNext(true);
    }

    var sessionService = new Mock<ITabStateService<ViewportSession>>();
    var sessionList = new[] { new SessionId(typeof(ViewportSession), 1) };
    sessionService.Setup(s => s.ActiveSessionIds).Returns(sessionList);
    sessionService.Setup(s => s.ObserveSessionList()).Returns(System.Reactive.Linq.Observable.Return<System.Collections.Generic.IReadOnlyList<SessionId>>(sessionList));
    sessionService.Setup(s => s.ObserveSession(It.IsAny<SessionId>())).Returns(System.Reactive.Linq.Observable.Return(new ViewportSession()));
    sessionService.Setup(s => s.GetSession(It.IsAny<SessionId>())).Returns(new ViewportSession());

    VulkanViewportControlViewModel VulkanFactory(Viewport3DViewModel vm) =>
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
        new BreadcrumbService(dispatcher.Object),
        dispatcher.Object,
        new Mock<IFileDialogService>().Object,
        vm
      ).Object;

    var platformWindowService = new Mock<IPlatformWindowService>().Object;

    return new Viewport3DViewModel(
      runtime.Object,
      breadcrumb,
      dispatcher.Object,
      new Mock<IFileDialogService>().Object,
      cameraService,
      VulkanFactory,
      OverlayFactory,
      platformWindowService,
      new Mock<IWindowInputRouter>().Object,
      sessionService.Object,
      new Mock<IViewportRegistry>().Object
    );
  }

  private static InputState Pressed(InputModifiers mods = InputModifiers.None)
    => new InputState(isPressed: true, mods);

  // ── Tests ────────────────────────────────────────────────────────────────────

  [Fact]
  public void Initialization_DefaultDimensions()
  {
    try
    {
      var vm = BuildVm();
      Assert.Equal(800u, vm.Width);
      Assert.Equal(600u, vm.Height);
      vm.Stop();
    }
    catch (TypeInitializationException) { /* native DLL absent in CI */ }
    catch (DllNotFoundException)        { /* native DLL absent in CI */ }
  }

  [Fact]
  public void SwitchCameraMode_V_CyclesThrough3Modes()
  {
    try
    {
      var vm = BuildVm(cometCommitted: true);
      // Initial state is UpZenith (EarthPosition requires SPK data)
      Assert.Equal(CameraMode.UpZenith, vm.CameraService.CurrentMode);

      vm.Process(new AppAction(ViewportAction.SwitchCameraMode.ToCmdString()), Pressed());
      Assert.Equal(CameraMode.CometOrbiting, vm.CameraService.CurrentMode);

      vm.Process(new AppAction(ViewportAction.SwitchCameraMode.ToCmdString()), Pressed());
      Assert.Equal(CameraMode.EarthPosition, vm.CameraService.CurrentMode);

      vm.Process(new AppAction(ViewportAction.SwitchCameraMode.ToCmdString()), Pressed());
      Assert.Equal(CameraMode.UpZenith, vm.CameraService.CurrentMode);
    }
    catch (TypeInitializationException) { }
    catch (DllNotFoundException)        { }
  }

  [Fact]
  public void SwitchToEarthPosition_DirectJump()
  {
    try
    {
      var vm = BuildVm(cometCommitted: true);
      vm.Process(new AppAction(ViewportAction.SwitchToCometOrbiting.ToCmdString()), Pressed());
      Assert.Equal(CameraMode.CometOrbiting, vm.CameraService.CurrentMode);

      vm.Process(new AppAction(ViewportAction.SwitchToEarthPosition.ToCmdString()), Pressed());
      Assert.Equal(CameraMode.EarthPosition, vm.CameraService.CurrentMode);
    }
    catch (TypeInitializationException) { }
    catch (DllNotFoundException)        { }
  }

  [Fact]
  public void SwitchToCometOrbiting_DirectJump()
  {
    try
    {
      var vm = BuildVm(cometCommitted: true);
      vm.Process(new AppAction(ViewportAction.SwitchToCometOrbiting.ToCmdString()), Pressed());
      Assert.Equal(CameraMode.CometOrbiting, vm.CameraService.CurrentMode);
    }
    catch (TypeInitializationException) { }
    catch (DllNotFoundException)        { }
  }

  [Fact]
  public void IsEarthObserverMode_TrueOnlyForEarthPosition()
  {
    try
    {
      var vm = BuildVm(cometCommitted: true);

      vm.Process(new AppAction(ViewportAction.SwitchToEarthPosition.ToCmdString()), Pressed());
      Assert.True(vm.IsEarthObserverMode);
      Assert.Equal(EarthObserverState.EarthPositioning, vm.EarthObserverState);

      vm.Process(new AppAction(ViewportAction.SwitchToUpZenith.ToCmdString()), Pressed());
      Assert.False(vm.IsEarthObserverMode);  // UpZenith is NOT EarthObserver
      Assert.Equal(EarthObserverState.UpZenith, vm.EarthObserverState);

      vm.Process(new AppAction(ViewportAction.SwitchToCometOrbiting.ToCmdString()), Pressed());
      Assert.False(vm.IsEarthObserverMode);
      Assert.Equal(EarthObserverState.CometOrbiting, vm.EarthObserverState);
    }
    catch (TypeInitializationException) { }
    catch (DllNotFoundException)        { }
  }

  [Fact]
  public void StartOrbit_PushesOperator_PointerDeltaHandled_ThenPopsOnRelease()
  {
    try
    {
      var vm = BuildVm();

      // EarthPosition required so IsOrbitAllowed() returns true
      vm.Process(new AppAction(ViewportAction.SwitchToEarthPosition.ToCmdString()), Pressed());

      bool startHandled = vm.Process(
        new AppAction(ViewportAction.StartOrbit.ToCmdString(), new Vector2(0, 0)),
        Pressed());
      Assert.True(startHandled);

      // OrbitCameraOperator consumes pointer_delta
      bool deltaHandled = vm.Process(
        new AppAction("viewport.pointer_delta", new Vector2(10, 10)),
        Pressed());
      Assert.True(deltaHandled);

      // Release pops the operator
      vm.Process(
        new AppAction("viewport.pointer_end"),
        new InputState(isPressed: false, InputModifiers.None));

      // Base operator ignores pointer_delta
      bool fallthrough = vm.Process(
        new AppAction("viewport.pointer_delta", new Vector2(10, 10)),
        Pressed());
      Assert.False(fallthrough);
    }
    catch (TypeInitializationException) { }
    catch (DllNotFoundException)        { }
  }
}
