using System;
using System.Numerics;
using AetherVk.Logic.Input;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using AetherVk.UI;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using Avalonia.VisualTree;

namespace AetherVk.Views;

public partial class Viewport3DView : UserControl
{
  private Viewport3DViewModel? _viewModel;

  // Overlay lifecycle refs — NOT parent-window caches.
  // Stored only to clean up what we created; walked fresh on every re-attach.
  private OverlayWindow?       _overlay;
  private OverlaySynchronizer? _synchronizer;
  private IWindowInputRouter?  _routerRef;

  public Viewport3DView()
  {
    InitializeComponent();
  }

  protected override void OnDataContextChanged(EventArgs e)
  {
    base.OnDataContextChanged(e);

    var newVm = DataContext as Viewport3DViewModel;

    // If the DataContext is being swapped while already attached (edge case: tab reuse),
    // tear down the old overlay first so we don't leak it.
    if (_viewModel != newVm && _overlay != null)
      TeardownOverlay();

    _viewModel = newVm;

    Console.WriteLine($"[Viewport3DView] OnDataContextChanged: newVm={newVm != null}, TopLevel={TopLevel.GetTopLevel(this)?.GetType().Name ?? "null"}");

    // DataContext normally arrives AFTER OnAttachedToVisualTree when the view is
    // instantiated by ViewLocator / DataTemplate (Avalonia propagates DataContext
    // top-down after inserting the child into the visual tree).
    // Call TrySetupOverlay here so the overlay is created even in that common case.
    // TrySetupOverlay is idempotent — it no-ops if the overlay already exists.
    if (TopLevel.GetTopLevel(this) != null)
      TrySetupOverlay();
  }

  protected override void OnAttachedToVisualTree(VisualTreeAttachmentEventArgs e)
  {
    base.OnAttachedToVisualTree(e);
    AddHandler(PointerPressedEvent,  OnViewportPointerPressed,  RoutingStrategies.Tunnel);
    AddHandler(PointerMovedEvent,    OnViewportPointerMoved,    RoutingStrategies.Tunnel);
    AddHandler(PointerReleasedEvent, OnViewportPointerReleased, RoutingStrategies.Tunnel);
    // Scroll wheel intentionally not attached — zoom uses Ctrl+Middle-Mouse drag only.

    Console.WriteLine($"[Viewport3DView] OnAttachedToVisualTree: _viewModel={_viewModel != null}, TopLevel={TopLevel.GetTopLevel(this)?.GetType().Name ?? "null"}");

    // ── Overlay Window ─────────────────────────────────────────────────────────
    // If DataContext is already set (unusual but possible if a view is re-inserted),
    // set up the overlay immediately. Otherwise OnDataContextChanged will call this
    // once DataContext is propagated — TrySetupOverlay is idempotent either way.
    TrySetupOverlay();
  }

  protected override void OnDetachedFromVisualTree(VisualTreeAttachmentEventArgs e)
  {
    TeardownOverlay();

    RemoveHandler(PointerPressedEvent,  OnViewportPointerPressed);
    RemoveHandler(PointerMovedEvent,    OnViewportPointerMoved);
    RemoveHandler(PointerReleasedEvent, OnViewportPointerReleased);
    base.OnDetachedFromVisualTree(e);
  }

  // ── Overlay lifecycle helpers ─────────────────────────────────────────────

  /// <summary>
  /// Creates and wires the <see cref="OverlayWindow"/> when both preconditions are met:
  /// the view is attached to a visual tree (so <c>TopLevel</c> is reachable) and
  /// <c>_viewModel</c> has been set via <c>DataContext</c>.
  /// Idempotent — subsequent calls are no-ops once <c>_overlay</c> is non-null.
  /// </summary>
  private void TrySetupOverlay()
  {
    Console.WriteLine($"[Viewport3DView] TrySetupOverlay called: _overlay={_overlay != null}, _viewModel={_viewModel != null}, TopLevel={TopLevel.GetTopLevel(this)?.GetType().Name ?? "null"}, ViewportHost={ViewportHost != null}");

    // Idempotent guard — do nothing if already set up or either precondition is missing.
    if (_overlay != null) { Console.WriteLine("[Viewport3DView] TrySetupOverlay: already set up"); return; }
    if (_viewModel == null) { Console.WriteLine("[Viewport3DView] TrySetupOverlay: _viewModel is null — waiting for DataContext"); return; }

    var mainWindow = TopLevel.GetTopLevel(this) as Window;
    if (mainWindow == null) { Console.WriteLine($"[Viewport3DView] TrySetupOverlay: TopLevel is not a Window (type={TopLevel.GetTopLevel(this)?.GetType().Name ?? "null"})"); return; }

    Console.WriteLine("[Viewport3DView] TrySetupOverlay: creating OverlayWindow…");
    try
    {
      _routerRef    = _viewModel.InputRouter;
      _overlay      = new OverlayWindow { DataContext = _viewModel.OverlayViewModel };
      _synchronizer = new OverlaySynchronizer(mainWindow, _overlay, ViewportHost, _viewModel.PlatformWindowService);
      _routerRef.AttachToWindow(_overlay);

      // ── Overlay pointer pass-through ───────────────────────────────────────
      // The overlay Avalonia window sits on top of the viewport XID in X11 stacking order
      // and intercepts ALL pointer events before they reach the native XID.
      // We subscribe here and relay transparent-area events so camera operators receive
      // the same input as if the overlay weren't there.
      _overlay.AddHandler(PointerPressedEvent,  OnOverlayPassThrough_Pressed,  RoutingStrategies.Tunnel);
      _overlay.AddHandler(PointerMovedEvent,    OnOverlayPassThrough_Moved,    RoutingStrategies.Tunnel);
      _overlay.AddHandler(PointerReleasedEvent, OnOverlayPassThrough_Released, RoutingStrategies.Tunnel);
      Console.WriteLine("[Viewport3DView] TrySetupOverlay: OverlayWindow created and wired OK");
    }
    catch (Exception ex)
    {
      Console.WriteLine($"[Viewport3DView] TrySetupOverlay: EXCEPTION — {ex}");
      _overlay      = null;
      _synchronizer = null;
      _routerRef    = null;
    }
  }

  /// <summary>
  /// Removes overlay event handlers, detaches the router, disposes the synchronizer,
  /// and closes the overlay window. Safe to call when <c>_overlay</c> is null.
  /// </summary>
  private void TeardownOverlay()
  {
    // Unsubscribe overlay pass-through BEFORE nulling _overlay.
    if (_overlay != null)
    {
      _overlay.RemoveHandler(PointerPressedEvent,  OnOverlayPassThrough_Pressed);
      _overlay.RemoveHandler(PointerMovedEvent,    OnOverlayPassThrough_Moved);
      _overlay.RemoveHandler(PointerReleasedEvent, OnOverlayPassThrough_Released);
    }

    // Detach overlay from router BEFORE closing it.
    if (_overlay != null && _routerRef != null)
      _routerRef.DetachFromWindow(_overlay);

    _synchronizer?.Dispose(); // unsubscribes bounds/position handlers, calls overlay.Close()
    _synchronizer = null;
    _overlay      = null;
    _routerRef    = null;
  }

  // ── Overlay pass-through ──────────────────────────────────────────────────
  // GlobalInputRouter returns early (no action) for TransparentRoot, which is correct.
  // But we also need to relay the event to the camera operators and request X11 focus.

  private void OnOverlayPassThrough_Pressed(object? sender, PointerPressedEventArgs e)
  {
    if (!IsTransparentVoid(e)) return;

    // Request X11 keyboard focus on the viewport native window so subsequent key events
    // (camera mode, shortcuts) reach the native input handler. No OS PInvoke here —
    // FocusViewport() calls through INativeInputHandler.FocusViewportWindow().
    _viewModel?.VulkanViewModel.FocusViewport();

    if (_viewModel is null) return;
    var pt   = e.GetCurrentPoint(sender as Visual);
    var pos  = new Vector2((float)pt.Position.X, (float)pt.Position.Y);
    var mods = GetModifiers(e.KeyModifiers);
    
    Console.WriteLine($"[OverlayPassThrough] Pressed! pt.Properties: L={pt.Properties.IsLeftButtonPressed}, M={pt.Properties.IsMiddleButtonPressed}, R={pt.Properties.IsRightButtonPressed}, pointerUpdate={pt.Properties.PointerUpdateKind}");

    if (pt.Properties.IsMiddleButtonPressed)
    {
      // Do NOT call e.Pointer.Capture(this): the event originates in the overlay window and
      // this is Viewport3DView in the main window. Cross-window Avalonia capture silently
      // fails. The overlay will continue to receive PointerMoved naturally since it is the
      // topmost window — OnOverlayPassThrough_Moved handles the drag below.
      string actionId =
        mods.HasFlag(InputModifiers.Ctrl)  ? ViewportAction.StartZoom.ToCmdString()
        : mods.HasFlag(InputModifiers.Shift) ? ViewportAction.StartPan.ToCmdString()
        : ViewportAction.StartOrbit.ToCmdString();

      Console.WriteLine($"[OverlayPassThrough] → dispatching actionId={actionId} pos={pos}");

      _viewModel.Process(new AppAction(actionId, pos), new InputState(isPressed: true, mods));
      e.Handled = true;
    }
  }

  private void OnOverlayPassThrough_Moved(object? sender, PointerEventArgs e)
  {
    if (_viewModel is null) return;
    var pt  = e.GetCurrentPoint(sender as Visual);
    var pos = new Vector2((float)pt.Position.X, (float)pt.Position.Y);

    // Forward whenever ANY mouse button is held (active drag) OR the pointer is over the
    // transparent void. The IsTransparentVoid-only check broke drags that crossed over
    // non-transparent widgets (badge, radial hub, etc.) mid-drag.
    bool anyButtonDown = pt.Properties.IsLeftButtonPressed
                      || pt.Properties.IsMiddleButtonPressed
                      || pt.Properties.IsRightButtonPressed;
    if (!anyButtonDown && !IsTransparentVoid(e)) return;

    var mods = GetModifiers(e.KeyModifiers);
    _viewModel.Process(
      new AppAction("viewport.pointer_delta", pos),
      new InputState(isPressed: anyButtonDown, mods));

    var overlayVm = _viewModel.OverlayViewModel;
    if (overlayVm.IsRadialMenuOpen)
      overlayVm.UpdateRadialMenuHover(pt.Position.X, pt.Position.Y);
  }

  private void OnOverlayPassThrough_Released(object? sender, PointerReleasedEventArgs e)
  {
    if (_viewModel is null) return;
    // Always end the camera operation — releasing a button over a widget must still
    // terminate the drag that was started on the transparent void.
    e.Pointer.Capture(null);
    _viewModel.Process(
      new AppAction("viewport.pointer_end"),
      new InputState(isPressed: false, GetModifiers(e.KeyModifiers)));
  }

  /// <summary>
  /// True when the event source is the transparent background of the overlay (no interactive
  /// UI widget was hit). Mirrors the check in <see cref="Input.GlobalInputRouter"/>.
  /// </summary>
  private static bool IsTransparentVoid(PointerEventArgs e)
    => e.Source is Window
    || (e.Source is Control c && c.Name == "TransparentRoot");

  // ── Viewport pointer handlers ─────────────────────────────────────────────

  private void OnViewportPointerPressed(object? sender, PointerPressedEventArgs e)
  {
    if (_viewModel is null) return;
    var pt   = e.GetCurrentPoint(this);
    var pos  = new Vector2((float)pt.Position.X, (float)pt.Position.Y);
    var mods = GetModifiers(e.KeyModifiers);

    if (pt.Properties.IsMiddleButtonPressed)
    {
      e.Pointer.Capture(this);
      // Ctrl → Zoom | Shift → Pan | (none) → Orbit
      string actionId =
        mods.HasFlag(InputModifiers.Ctrl)  ? ViewportAction.StartZoom.ToCmdString()
        : mods.HasFlag(InputModifiers.Shift) ? ViewportAction.StartPan.ToCmdString()
        : ViewportAction.StartOrbit.ToCmdString();

      _viewModel.Process(new AppAction(actionId, pos), new InputState(isPressed: true, mods));
      e.Handled = true;
    }
  }

  private void OnViewportPointerMoved(object? sender, PointerEventArgs e)
  {
    if (_viewModel is null) return;
    var pt   = e.GetCurrentPoint(this);
    var pos  = new Vector2((float)pt.Position.X, (float)pt.Position.Y);
    var mods = GetModifiers(e.KeyModifiers);

    _viewModel.Process(
      new AppAction("viewport.pointer_delta", pos),
      new InputState(isPressed: true, mods));

    var overlayVm = _viewModel.OverlayViewModel;
    if (overlayVm.IsRadialMenuOpen)
      overlayVm.UpdateRadialMenuHover(pt.Position.X, pt.Position.Y);
  }

  private void OnViewportPointerReleased(object? sender, PointerReleasedEventArgs e)
  {
    if (_viewModel is null) return;
    e.Pointer.Capture(null);
    _viewModel.Process(
      new AppAction("viewport.pointer_end"),
      new InputState(isPressed: false, GetModifiers(e.KeyModifiers)));
  }

  private static InputModifiers GetModifiers(KeyModifiers km)
  {
    var m = InputModifiers.None;
    if (km.HasFlag(KeyModifiers.Shift))   m |= InputModifiers.Shift;
    if (km.HasFlag(KeyModifiers.Control)) m |= InputModifiers.Ctrl;
    if (km.HasFlag(KeyModifiers.Alt))     m |= InputModifiers.Alt;
    return m;
  }
}
