using AetherVk.Input;
using AetherVk.Logic.Input;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Input;
using Avalonia;
using Moq;
using Xunit;

namespace AetherVk.AppTests;

/// <summary>
/// Integration tests for <see cref="GlobalInputRouter"/> using the Avalonia headless platform.
/// Each test constructs a minimal visual tree, attaches the router to a real headless
/// <see cref="Window"/>, simulates keyboard events via <see cref="HeadlessWindowExtensions"/>,
/// and asserts that the correct <see cref="AppAction"/> reaches the mock handler.
///
/// These tests cover the complete pipeline:
///   KeyPressQwerty → OnKeyDown → HandleInput → RouteAction (visual tree walk) → handler.Process()
/// </summary>
public class GlobalInputRouterTests
{
  // ── Helpers ────────────────────────────────────────────────────────────────

  /// <summary>
  /// Builds a headless <see cref="Window"/> containing a single focusable <see cref="Border"/>
  /// tagged with <paramref name="contextId"/> and <paramref name="handler"/> via
  /// <see cref="ActionContext"/> attached properties.
  /// </summary>
  private static (Window Window, Border ContextBorder) BuildWindow(
    string contextId,
    IActionHandler handler,
    Control? innerContent = null)
  {
    var border = new Border
    {
      Focusable = true,
      Width = 200,
      Height = 200,
      Child = innerContent,
    };
    ActionContext.SetId(border, contextId);
    ActionContext.SetHandler(border, handler);

    var window = new Window { Content = border, Width = 400, Height = 300 };
    return (window, border);
  }

  private static IActionHandler MockHandler(out Mock<IActionHandler> mock)
  {
    mock = new Mock<IActionHandler>();
    mock.Setup(h => h.Process(It.IsAny<AppAction>(), It.IsAny<InputState>()))
        .Returns(true);
    return mock.Object;
  }

  // ── Tests ──────────────────────────────────────────────────────────────────

  [AvaloniaFact]
  public void KeyPress_RegisteredChord_DispatchesToHandler()
  {
    // Arrange
    var registry = new InputRegistry();
    registry.Register("Viewport",
      new InputChord(Key: "V"),
      new AppAction("viewport.switch_camera_mode"));

    var router = new GlobalInputRouter(registry);

    var handler = MockHandler(out var mock);
    var (window, border) = BuildWindow("Viewport", handler);
    window.Show();
    router.AttachToWindow(window);
    border.Focus();

    // Act — KeyPressQwerty is the non-deprecated overload; translates PhysicalKey → logical Key
    // via QWERTY layout map. PhysicalKey.V → Key.V.
    window.KeyPressQwerty(PhysicalKey.V, RawInputModifiers.None);

    // Assert: handler called once with pressed=true
    mock.Verify(h =>
      h.Process(
        It.Is<AppAction>(a => a.Id == "viewport.switch_camera_mode"),
        It.Is<InputState>(s => s.IsPressed)),
      Times.Once);

    router.Dispose();
    window.Close();
  }

  [AvaloniaFact]
  public void KeyRelease_RegisteredChord_DispatchesWithIsPressed_False()
  {
    // Arrange
    var registry = new InputRegistry();
    registry.Register("Viewport",
      new InputChord(Key: "V"),
      new AppAction("viewport.switch_camera_mode"));

    var router = new GlobalInputRouter(registry);

    var handler = MockHandler(out var mock);
    var (window, border) = BuildWindow("Viewport", handler);
    window.Show();
    router.AttachToWindow(window);
    border.Focus();

    // Act — press then release
    window.KeyPressQwerty(PhysicalKey.V, RawInputModifiers.None);
    window.KeyReleaseQwerty(PhysicalKey.V, RawInputModifiers.None);

    // Assert: called for both press and release
    mock.Verify(h =>
      h.Process(
        It.Is<AppAction>(a => a.Id == "viewport.switch_camera_mode"),
        It.Is<InputState>(s => s.IsPressed)),
      Times.Once, "key-down dispatch");

    mock.Verify(h =>
      h.Process(
        It.Is<AppAction>(a => a.Id == "viewport.switch_camera_mode"),
        It.Is<InputState>(s => !s.IsPressed)),
      Times.Once, "key-up dispatch");

    router.Dispose();
    window.Close();
  }

  [AvaloniaFact]
  public void KeyPress_UnregisteredChord_DoesNotDispatch()
  {
    // Arrange — empty registry, no bindings
    var registry = new InputRegistry();
    var router = new GlobalInputRouter(registry);

    var handler = MockHandler(out var mock);
    var (window, border) = BuildWindow("Viewport", handler);
    window.Show();
    router.AttachToWindow(window);
    border.Focus();

    // Act — Z is not registered
    window.KeyPressQwerty(PhysicalKey.Z, RawInputModifiers.None);

    // Assert: handler must NOT have been called
    mock.Verify(h => h.Process(It.IsAny<AppAction>(), It.IsAny<InputState>()), Times.Never);

    router.Dispose();
    window.Close();
  }

  [AvaloniaFact]
  public void KeyPress_NestedVisualTree_WalksUpToContextBorder()
  {
    // Arrange — focused element is a TextBlock nested INSIDE the border, not the border itself.
    // The router must walk up the visual tree and discover ActionContext on the border ancestor.
    var registry = new InputRegistry();
    registry.Register("Viewport",
      new InputChord(Key: "NumPad1"),
      new AppAction("viewport.switch_to_earth_position"));

    var router = new GlobalInputRouter(registry);

    var innerLabel = new TextBlock { Text = "inner", Focusable = true };
    var handler = MockHandler(out var mock);
    var (window, border) = BuildWindow("Viewport", handler, innerContent: innerLabel);
    window.Show();
    router.AttachToWindow(window);

    // Focus the inner child — router must walk up to the border to find the context
    innerLabel.Focus();

    // Act
    window.KeyPressQwerty(PhysicalKey.NumPad1, RawInputModifiers.None);

    // Assert
    mock.Verify(h =>
      h.Process(
        It.Is<AppAction>(a => a.Id == "viewport.switch_to_earth_position"),
        It.Is<InputState>(s => s.IsPressed)),
      Times.Once);

    router.Dispose();
    window.Close();
  }

  [AvaloniaFact]
  public void KeyPress_WithModifier_OnlyMatchesCorrectChord()
  {
    // Arrange — register Ctrl+Z only; plain Z must NOT match
    var registry = new InputRegistry();
    registry.Register("Viewport",
      new InputChord(Key: "Z", Ctrl: true),
      new AppAction("viewport.undo"));

    var router = new GlobalInputRouter(registry);
    var handler = MockHandler(out var mock);
    var (window, border) = BuildWindow("Viewport", handler);
    window.Show();
    router.AttachToWindow(window);
    border.Focus();

    // Act 1: plain Z — should NOT match Ctrl+Z
    window.KeyPressQwerty(PhysicalKey.Z, RawInputModifiers.None);
    mock.Verify(h => h.Process(It.IsAny<AppAction>(), It.IsAny<InputState>()), Times.Never,
      "plain Z must not match Ctrl+Z binding");

    // Act 2: Ctrl+Z — should match
    window.KeyPressQwerty(PhysicalKey.Z, RawInputModifiers.Control);
    mock.Verify(h =>
      h.Process(
        It.Is<AppAction>(a => a.Id == "viewport.undo"),
        It.Is<InputState>(s => s.IsPressed && s.Modifiers.HasFlag(Logic.Input.InputModifiers.Ctrl))),
      Times.Once);

    router.Dispose();
    window.Close();
  }

  [AvaloniaFact]
  public void DetachFromWindow_StopsDispatching()
  {
    // Arrange
    var registry = new InputRegistry();
    registry.Register("Viewport",
      new InputChord(Key: "V"),
      new AppAction("viewport.switch_camera_mode"));

    var router = new GlobalInputRouter(registry);
    var handler = MockHandler(out var mock);
    var (window, border) = BuildWindow("Viewport", handler);
    window.Show();
    router.AttachToWindow(window);
    border.Focus();

    // Act 1 — while attached
    window.KeyPressQwerty(PhysicalKey.V, RawInputModifiers.None);
    mock.Verify(h => h.Process(It.IsAny<AppAction>(), It.IsAny<InputState>()), Times.Once);

    // Detach
    router.DetachFromWindow(window);

    // Act 2 — after detach: second press must NOT produce a second dispatch
    window.KeyPressQwerty(PhysicalKey.V, RawInputModifiers.None);
    mock.Verify(h => h.Process(It.IsAny<AppAction>(), It.IsAny<InputState>()), Times.Once,
      "no new dispatch expected after DetachFromWindow");

    router.Dispose();
    window.Close();
  }

  // ── Keys with no focus after a click in a native viewport (macOS) ──────────
  // On macOS the native viewport never gets keyDown: (Avalonia's window keeps the key events), so
  // after a click in it the keys reach Avalonia with nothing focused and were dropped.

  private static (GlobalInputRouter Router, Window Window, Mock<IActionHandler> Mock, Button Button) UnfocusedViewport()
  {
    var registry = new InputRegistry();
    registry.Register("Viewport", new InputChord(Key: "D3"), new AppAction("viewport.switch_to_comet_orbiting"));
    var router = new GlobalInputRouter(registry);
    var handler = MockHandler(out var mock);
    var button = new Button { Content = "panel", Width = 50, Height = 30 };
    var (window, _) = BuildWindow("Viewport", handler, button);
    window.Show();
    router.AttachToWindow(window);
    window.FocusManager?.ClearFocus();
    return (router, window, mock, button);
  }

  private static void VerifyCometOrbiting(Mock<IActionHandler> mock, Times times) =>
    mock.Verify(h => h.Process(It.Is<AppAction>(a => a.Id == "viewport.switch_to_comet_orbiting"), It.IsAny<InputState>()), times);

  [AvaloniaFact]
  public void UnfocusedKey_AfterANativeViewportClick_GoesToTheViewport_OnPressOnly()
  {
    var (router, window, mock, _) = UnfocusedViewport();
    router.NativeViewportPressed();

    window.KeyPressQwerty(PhysicalKey.Digit3, RawInputModifiers.None);
    window.KeyReleaseQwerty(PhysicalKey.Digit3, RawInputModifiers.None);

    mock.Verify(h => h.Process(It.Is<AppAction>(a => a.Id == "viewport.switch_to_comet_orbiting"),
      It.Is<InputState>(st => st.IsPressed)), Times.Once);
    VerifyCometOrbiting(mock, Times.Once());
    router.Dispose();
    window.Close();
  }

  [AvaloniaFact]
  public void UnfocusedKey_WithoutANativeViewportClick_IsDropped()
  {
    var (router, window, mock, _) = UnfocusedViewport();

    window.KeyPressQwerty(PhysicalKey.Digit3, RawInputModifiers.None);

    VerifyCometOrbiting(mock, Times.Never());
    router.Dispose();
    window.Close();
  }

  [AvaloniaFact]
  public void UnfocusedKey_AfterAClickOnAnAvaloniaControl_IsDropped()
  {
    var (router, window, mock, button) = UnfocusedViewport();
    router.NativeViewportPressed();
    window.MouseDown(button.TranslatePoint(new Avalonia.Point(5, 5), window)!.Value, MouseButton.Left);
    window.MouseUp(button.TranslatePoint(new Avalonia.Point(5, 5), window)!.Value, MouseButton.Left);
    window.FocusManager?.ClearFocus();

    window.KeyPressQwerty(PhysicalKey.Digit3, RawInputModifiers.None);

    VerifyCometOrbiting(mock, Times.Never());
    router.Dispose();
    window.Close();
  }

  // ── A desktop billboard over the viewport (Windows: its window takes the clicks) ──

  /// <summary>A billboard window: a canvas with a focusable image, as DesktopBillboardWindow.</summary>
  private static (Window Window, Border Image) Billboard()
  {
    var image = new Border { Focusable = true, Width = 100, Height = 100, Background = Avalonia.Media.Brushes.Gray };
    var window = new Window { Content = new Canvas { Children = { image } }, Width = 120, Height = 120 };
    window.Show();
    return (window, image);
  }

  [AvaloniaFact]
  public void KeyOnABillboard_AfterAClickOnIt_GoesToTheViewportBelow()
  {
    var (router, window, mock, _) = UnfocusedViewport();
    var (billboard, image) = Billboard();
    router.AttachViewportOverlay(billboard);
    billboard.MouseDown(new Avalonia.Point(50, 50), MouseButton.Left);
    billboard.MouseUp(new Avalonia.Point(50, 50), MouseButton.Left);
    image.Focus();

    billboard.KeyPressQwerty(PhysicalKey.Digit3, RawInputModifiers.None);
    billboard.KeyReleaseQwerty(PhysicalKey.Digit3, RawInputModifiers.None);

    VerifyCometOrbiting(mock, Times.Once());
    router.Dispose();
    billboard.Close();
    window.Close();
  }

  [AvaloniaFact]
  public void KeyOnABillboard_AfterItIsDetached_IsNotRouted()
  {
    var (router, window, mock, _) = UnfocusedViewport();
    var (billboard, _) = Billboard();
    router.AttachViewportOverlay(billboard);
    router.DetachFromWindow(billboard);
    billboard.MouseDown(new Avalonia.Point(50, 50), MouseButton.Left);
    billboard.MouseUp(new Avalonia.Point(50, 50), MouseButton.Left);

    billboard.KeyPressQwerty(PhysicalKey.Digit3, RawInputModifiers.None);

    VerifyCometOrbiting(mock, Times.Never());
    router.Dispose();
    billboard.Close();
    window.Close();
  }
}
