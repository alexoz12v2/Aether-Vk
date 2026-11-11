using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive.Disposables;
using AetherVk.Logic.Input;
using AetherVk.Logic.Services;
using AetherVk.Logic.Services.NativeInput;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The "Nucleus seen from Earth" view of the compact panel, against a mocked engine: its native
/// view is made once, on the first show; nothing is drawn before the camera's flags and the view
/// from Earth are set; it pauses when collapsed, in the classic layout or minimized and resumes
/// when shown; a second viewport is never added; the expander's state is remembered.
/// </summary>
public sealed class NucleusMiniViewModelTests : IDisposable
{
  private const ulong Pe = 20, Camera = 21, Earth = 3, Comet = 4;

  private readonly string _root = Path.Combine(Path.GetTempPath(), "avk_nucleusmini_" + Guid.NewGuid().ToString("N"));
  private readonly Mock<INativeRuntimeService> _runtime = new();
  private int _viewportsMade;

  public NucleusMiniViewModelTests()
  {
    Directory.CreateDirectory(_root);
    ulong outPe = Pe, outCam = Camera;
    _runtime.Setup(r => r.AddViewport(
        It.IsAny<uint>(), It.IsAny<uint>(), It.IsAny<string>(), It.IsAny<Func<CNativeWindowHandle>?>(),
        It.IsAny<uint>(), out outPe, out outCam, It.IsAny<ViewportRole>()))
      .Returns(true);
    _runtime.Setup(r => r.EarthEntityId).Returns(Earth);
    _runtime.Setup(r => r.CometEntityId).Returns(Comet);
    _runtime.Setup(r => r.SetCameraNoDust(Camera, true)).Returns(true);
    _runtime.Setup(r => r.SetCameraNoUiLayer(Camera, true)).Returns(true);
    _runtime.Setup(r => r.SetCameraShading(Camera, It.IsAny<uint>())).Returns(true);
    _runtime.Setup(r => r.SetCameraParentToComet(Camera, true)).Returns(true);
    _runtime.Setup(r => r.CameraSetPerspective(Camera, It.IsAny<float>(), It.IsAny<float>(), It.IsAny<float>(), It.IsAny<float>())).Returns(true);
    _runtime.Setup(r => r.SetNucleusViewFromEarth(Camera, Earth, Comet, It.IsAny<double>())).Returns(true);
    _runtime.Setup(r => r.SetViewportActive(Pe, It.IsAny<bool>())).Returns(true);
    _runtime.Setup(r => r.CameraSetRotoTranslate(Camera, It.IsAny<double>(), It.IsAny<double>(), It.IsAny<double>(), It.IsAny<Quaterniond>(), It.IsAny<ulong>())).Returns(true);
    _runtime.Setup(r => r.SetEarthObserver(Camera, -1, It.IsAny<ulong>(), It.IsAny<ulong>(), It.IsAny<bool>(), It.IsAny<double>(),
      It.IsAny<double>(), It.IsAny<double>(), It.IsAny<Quaterniond>(), It.IsAny<ObserverUp>())).Returns(true);
  }

  public void Dispose()
  {
    try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
  }

  private sealed class FakeHandler : INativeInputHandlerSubscribable
  {
    public Action<NativeMouseEvent>? Mouse;
    public int KeystrokeSubscriptions;
    public Action<ComposedKeystroke>? Keys;
    public void FocusViewportWindow() { }
    public void Dispose() { }

    public IDisposable SubscribeComposedKeystrokes(Action<ComposedKeystroke> onNext)
    {
      KeystrokeSubscriptions++;
      Keys = onNext;
      return Disposable.Empty;
    }

    public IDisposable SubscribeNativeMouseEvents(Action<NativeMouseEvent> onNext)
    {
      Mouse = onNext;
      return Disposable.Empty;
    }
  }

  private readonly FakeHandler _handler = new();
  private readonly Mock<IWindowInputRouter> _router = new();

  private PreferencesService Preferences()
  {
    var prefs = new PreferencesService(Path.Combine(_root, "preferences.json"), new InputRegistry());
    prefs.Load();
    return prefs;
  }

  private NucleusMiniViewModel Create(PreferencesService? prefs = null)
  {
    var handlers = new Mock<INativeInputHandlerFactory>();
    handlers.Setup(f => f.Create(It.IsAny<IntPtr>(), It.IsAny<string>(), It.IsAny<TraceLevel>())).Returns(_handler);
    return new NucleusMiniViewModel(
      _runtime.Object,
      pointer =>
      {
        _viewportsMade++;
        return new VulkanViewportControlViewModel(_router.Object, handlers.Object, _runtime.Object, null!)
        {
          Role = ViewportRole.NucleusMini,
          MiniPointer = pointer,
        };
      },
      translation: null,
      preferences: prefs);
  }

  /// <summary>What the native control does once its window exists.</summary>
  private static void CreateNativeWindow(NucleusMiniViewModel mini) =>
    Assert.True(mini.Viewport!.InitializeHandle(new IntPtr(0x4321), "XID"));

  private List<string> Calls() => _runtime.Invocations.Select(i => i.Method.Name).ToList();

  private List<bool> ActiveCalls() => _runtime.Invocations
    .Where(i => i.Method.Name == nameof(INativeRuntimeService.SetViewportActive))
    .Select(i => (bool)i.Arguments[1])
    .ToList();

  private void VerifyAddedOnce() =>
    _runtime.Verify(r => r.AddViewport(
      It.IsAny<uint>(), It.IsAny<uint>(), It.IsAny<string>(), It.IsAny<Func<CNativeWindowHandle>?>(),
      It.IsAny<uint>(), out It.Ref<ulong>.IsAny, out It.Ref<ulong>.IsAny, ViewportRole.NucleusMini), Times.Once);

  /// <summary>A view shown in the compact layout, its window created, a comet committed.</summary>
  private NucleusMiniViewModel Shown()
  {
    var mini = Create();
    mini.SetCometCommitted(true);
    mini.IsLayoutCompact = true;
    CreateNativeWindow(mini);
    Assert.True(mini.IsActive);
    return mini;
  }

  [Fact]
  public void HostSide_IsThePanelWidthLessTheGutter_AtMost320()
  {
    Assert.Equal(284.0, NucleusMiniViewModel.HostSide(300));
    Assert.Equal(320.0, NucleusMiniViewModel.HostSide(336));
    Assert.Equal(320.0, NucleusMiniViewModel.HostSide(900));
    Assert.Equal(0.0, NucleusMiniViewModel.HostSide(10));
    Assert.Equal(0.0, NucleusMiniViewModel.HostSide(double.NaN));
  }

  [Fact]
  public void TheNativeView_IsMadeOnTheFirstExpandedShowInCompact_AndOnlyOnce()
  {
    using var mini = Create();
    Assert.True(mini.IsExpanded); // the default
    Assert.Null(mini.Viewport);   // classic layout: nothing yet
    Assert.Equal(0, _viewportsMade);

    mini.IsLayoutCompact = true;
    Assert.NotNull(mini.Viewport);
    Assert.Equal(ViewportRole.NucleusMini, mini.Viewport!.Role);
    Assert.Equal(1, _viewportsMade);
    Assert.True(mini.IsHostVisible, "the host is shown so that its native window gets created");
    // the engine is attached by the native control, not by the view model
    Assert.DoesNotContain(nameof(INativeRuntimeService.AddViewport), Calls());

    var viewport = mini.Viewport;
    CreateNativeWindow(mini);
    VerifyAddedOnce();

    // every way of hiding and showing it again keeps that one view
    mini.IsExpanded = false;
    mini.IsExpanded = true;
    mini.IsLayoutCompact = false;
    mini.IsLayoutCompact = true;
    mini.IsWindowMinimized = true;
    mini.IsWindowMinimized = false;
    mini.Refresh();
    Assert.Same(viewport, mini.Viewport);
    Assert.Equal(1, _viewportsMade);
    VerifyAddedOnce();
    // and it is never removed by the view model (the shutdown removes every viewport)
    _runtime.Verify(r => r.RemoveViewport(It.IsAny<ulong>()), Times.Never);
  }

  [Fact]
  public void Collapsed_OrClassic_TheNativeViewIsNotMade()
  {
    var prefs = Preferences();
    prefs.SetNucleusMiniExpanded(false);
    using var mini = Create(prefs);
    Assert.False(mini.IsExpanded);
    mini.IsLayoutCompact = true;
    Assert.Null(mini.Viewport);
    Assert.Equal(0, _viewportsMade);

    mini.IsExpanded = true; // first shown expanded
    Assert.NotNull(mini.Viewport);
    Assert.Equal(1, _viewportsMade);
  }

  [Fact]
  public void NothingIsDrawn_BeforeTheFlagsAndTheViewFromEarthAreSet()
  {
    using var mini = Create();
    mini.SetCometCommitted(true);
    mini.IsLayoutCompact = true;
    CreateNativeWindow(mini);

    var calls = Calls();
    int active = calls.IndexOf(nameof(INativeRuntimeService.SetViewportActive));
    Assert.True(active >= 0, "the view was resumed");
    Assert.Equal(new[] { true }, ActiveCalls());
    foreach (var before in new[]
    {
      nameof(INativeRuntimeService.AddViewport),
      nameof(INativeRuntimeService.SetCameraNoDust),
      nameof(INativeRuntimeService.SetCameraNoUiLayer),
      nameof(INativeRuntimeService.SetCameraShading),
      nameof(INativeRuntimeService.SetCameraParentToComet),
      nameof(INativeRuntimeService.CameraSetPerspective),
      nameof(INativeRuntimeService.SetNucleusViewFromEarth),
      nameof(INativeRuntimeService.SetViewportMinFrameInterval),
    })
    {
      int at = calls.IndexOf(before);
      Assert.True(at >= 0, $"{before} was not called");
      Assert.True(at < active, $"{before} came after the view was resumed");
    }
    _runtime.Verify(r => r.SetCameraNoDust(Camera, true), Times.Once);
    _runtime.Verify(r => r.SetCameraNoUiLayer(Camera, true), Times.Once);
    // 5 fps: the main view shares the render thread
    _runtime.Verify(r => r.SetViewportMinFrameInterval(Pe, NucleusMiniViewModel.FrameIntervalUs), Times.Once);
    _runtime.Verify(r => r.SetCameraShading(Camera, CameraService.ShadingPhysical | CameraService.ShadingLambertDark), Times.Once);
    // this camera's own observer: the nucleus 4 diameters over a 20° field
    _runtime.Verify(r => r.SetNucleusViewFromEarth(Camera, Earth, Comet, mini.DistanceAu), Times.Once);
    Assert.True(mini.HasNucleusView);
    Assert.True(mini.IsEarthView);
    Assert.True(mini.IsHostVisible);
    Assert.False(mini.IsWaitingForComet);
  }

  [Fact]
  public void WithoutAComet_TheViewStaysPaused_UntilOneIsCommitted()
  {
    using var mini = Create();
    mini.IsLayoutCompact = true;
    CreateNativeWindow(mini);

    Assert.Empty(ActiveCalls()); // created paused, never resumed
    Assert.False(mini.IsActive);
    Assert.True(mini.IsWaitingForComet);
    Assert.False(mini.IsHostVisible);
    _runtime.Verify(r => r.SetNucleusViewFromEarth(It.IsAny<ulong>(), It.IsAny<ulong>(), It.IsAny<ulong>(), It.IsAny<double>()), Times.Never);

    mini.SetCometCommitted(true);
    Assert.Equal(new[] { true }, ActiveCalls());
    Assert.True(mini.IsActive);
    Assert.True(mini.IsHostVisible);

    // the comet is taken away: paused again, and the view from Earth is set again for the next one
    mini.SetCometCommitted(false);
    Assert.Equal(new[] { true, false }, ActiveCalls());
    mini.SetCometCommitted(true);
    Assert.Equal(new[] { true, false, true }, ActiveCalls());
    _runtime.Verify(r => r.SetNucleusViewFromEarth(Camera, Earth, Comet, It.IsAny<double>()), Times.Exactly(2));
  }

  [Fact]
  public void AFlagTheEngineRefuses_KeepsTheViewPaused()
  {
    _runtime.Setup(r => r.SetCameraNoDust(Camera, true)).Returns(false);
    using var mini = Create();
    mini.SetCometCommitted(true);
    mini.IsLayoutCompact = true;
    CreateNativeWindow(mini);
    Assert.Empty(ActiveCalls());

    // accepted on a later try
    _runtime.Setup(r => r.SetCameraNoDust(Camera, true)).Returns(true);
    mini.Refresh();
    Assert.Equal(new[] { true }, ActiveCalls());
  }

  [Fact]
  public void Collapse_Classic_Minimize_PauseIt_ShowingResumesIt()
  {
    using var mini = Shown();
    Assert.Equal(new[] { true }, ActiveCalls());

    mini.IsExpanded = false;
    Assert.Equal(new[] { true, false }, ActiveCalls());
    mini.IsExpanded = true;
    Assert.Equal(new[] { true, false, true }, ActiveCalls());

    mini.IsLayoutCompact = false;
    Assert.Equal(new[] { true, false, true, false }, ActiveCalls());
    mini.IsLayoutCompact = true;
    Assert.Equal(new[] { true, false, true, false, true }, ActiveCalls());

    mini.IsWindowMinimized = true;
    Assert.Equal(new[] { true, false, true, false, true, false }, ActiveCalls());
    mini.IsWindowMinimized = false;
    Assert.Equal(new[] { true, false, true, false, true, false, true }, ActiveCalls());

    // hidden twice over: one pause, and it resumes only when nothing hides it any more
    mini.IsExpanded = false;
    mini.IsLayoutCompact = false;
    mini.IsExpanded = true;
    Assert.Equal(8, ActiveCalls().Count);
    Assert.False(ActiveCalls().Last());
    mini.IsLayoutCompact = true;
    Assert.True(ActiveCalls().Last());

    _runtime.Verify(r => r.SetViewportActive(It.Is<ulong>(pe => pe != Pe), It.IsAny<bool>()), Times.Never);
    // the flags and the view from Earth were set once: pausing does not undo them
    _runtime.Verify(r => r.SetCameraNoDust(Camera, true), Times.Once);
    _runtime.Verify(r => r.SetNucleusViewFromEarth(Camera, Earth, Comet, It.IsAny<double>()), Times.Once);
    VerifyAddedOnce();
    Assert.Equal(1, _viewportsMade);
  }

  [Fact]
  public void ExpanderState_IsRememberedInThePreferences()
  {
    var prefs = Preferences();
    Assert.True(prefs.NucleusMiniExpanded); // nothing saved: expanded
    using (var mini = Create(prefs))
    {
      Assert.True(mini.IsExpanded);
      mini.IsExpanded = false;
    }
    var reloaded = Preferences();
    Assert.False(reloaded.NucleusMiniExpanded);
    using (var mini = Create(reloaded))
    {
      Assert.False(mini.IsExpanded);
      mini.IsExpanded = true;
    }
    Assert.True(Preferences().NucleusMiniExpanded);
  }

  [Fact]
  public void TheProjection_FollowsTheHostSize()
  {
    using var mini = Shown();
    _runtime.Invocations.Clear();
    mini.Viewport!.NotifyResized(300, 150);
    _runtime.Verify(r => r.CameraSetPerspective(Camera, NucleusMiniViewModel.FovRad, 2f, It.IsAny<float>(), It.IsAny<float>()), Times.Once);
    // the same size again: nothing is sent
    mini.Viewport.NotifyResized(300, 150);
    _runtime.Verify(r => r.CameraSetPerspective(It.IsAny<ulong>(), It.IsAny<float>(), It.IsAny<float>(), It.IsAny<float>(), It.IsAny<float>()), Times.Once);
  }

  [Fact]
  public void Dragging_OrbitsTheCamera_BackToEarthViewHandsItBack()
  {
    // the comet at RA 0, Dec 0 seen from Earth
    _runtime.Setup(r => r.CometEphemeris()).Returns(new CometEphemeris(DateTimeOffset.UnixEpoch, 0, 0, 1.2, 0.4, 0, 0, 0, 200));
    using var mini = Shown();
    Assert.False(mini.ResetToEarthCommand.CanExecute(null));
    Assert.NotNull(_handler.Mouse); // the view's own pointer
    _runtime.Invocations.Clear();

    // press, move 30 px to the right, release
    _handler.Mouse!(new NativeMouseEvent(100, 100, MouseButton.Left, true, default, 0));
    _runtime.Verify(r => r.CameraSetRotoTranslate(It.IsAny<ulong>(), It.IsAny<double>(), It.IsAny<double>(), It.IsAny<double>(),
      It.IsAny<Quaterniond>(), It.IsAny<ulong>()), Times.Never); // a press alone moves nothing
    _handler.Mouse(new NativeMouseEvent(130, 100, MouseButton.Left, true, default, 1));
    _handler.Mouse(new NativeMouseEvent(130, 100, MouseButton.Left, false, default, 2));

    Assert.False(mini.IsEarthView);
    // the engine stops posing this camera, then the view model poses it about the comet
    _runtime.Verify(r => r.SetEarthObserver(Camera, -1, It.IsAny<ulong>(), It.IsAny<ulong>(), It.IsAny<bool>(), It.IsAny<double>(),
      It.IsAny<double>(), It.IsAny<double>(), It.IsAny<Quaterniond>(), It.IsAny<ObserverUp>()), Times.Once);
    var pose = Assert.Single(_runtime.Invocations, i => i.Method.Name == nameof(INativeRuntimeService.CameraSetRotoTranslate));
    Assert.Equal(Camera, (ulong)pose.Arguments[0]);
    Assert.Equal(Comet, (ulong)pose.Arguments[5]);
    var offset = new Vector3d((double)pose.Arguments[1], (double)pose.Arguments[2], (double)pose.Arguments[3]);
    // still at the view's distance from the nucleus, turned by 30 px of drag from the Earth's side
    Assert.Equal(mini.DistanceAu, offset.Length(), 12);
    var fromEarth = -SkyMath.EclipticDirection(0, 0);
    double angle = Math.Acos(Math.Max(-1, Math.Min(1, Vector3d.Dot(Vector3d.Normalize(offset), fromEarth))));
    Assert.Equal(30 * NucleusMiniViewModel.OrbitRadPerPixel, angle, 9);

    // a move with no button held does nothing
    _handler.Mouse(new NativeMouseEvent(200, 180, MouseButton.None, false, default, 3));
    Assert.Single(_runtime.Invocations, i => i.Method.Name == nameof(INativeRuntimeService.CameraSetRotoTranslate));

    Assert.True(mini.ResetToEarthCommand.CanExecute(null));
    mini.ResetToEarthCommand.Execute(null);
    Assert.True(mini.IsEarthView);
    _runtime.Verify(r => r.SetNucleusViewFromEarth(Camera, Earth, Comet, mini.DistanceAu), Times.Once);
    Assert.Equal(mini.StrCaptionEarth, mini.CaptionText);
  }

  [Fact]
  public void Strings_AreEnglishWithoutATranslationService()
  {
    using var mini = Create();
    Assert.Equal("Nucleus seen from Earth", mini.StrHeader);
    Assert.Equal("Back to Earth view", mini.StrResetToEarth);
    Assert.Equal(mini.StrCaptionEarth, mini.CaptionText);
  }

  /// <summary>
  /// A key typed while the mini-view's native window has the focus (after a click or a drag on
  /// it) goes to the "Viewport" context, the main viewport's handler, and never moves the mini
  /// camera: it is neither dropped nor taken by the mini-view.
  /// </summary>
  [Fact]
  public void KeysTypedOnTheMini_GoToTheMainViewportContext()
  {
    using var mini = Shown();
    Assert.Equal(1, _handler.KeystrokeSubscriptions);
    _runtime.Invocations.Clear();
    _handler.Keys!(new ComposedKeystroke([new NativeKeyEvent(0x56, IsDown: true, NativeModifierFlags.None, 0)], 0));
    _router.Verify(r => r.RouteNativeComposed("Viewport", It.Is<InputChord>(c => c.Key == "V"), It.IsAny<InputState>()), Times.Once);
    Assert.DoesNotContain(nameof(INativeRuntimeService.CameraSetRotoTranslate), Calls());
    Assert.True(mini.IsEarthView);
  }
}
