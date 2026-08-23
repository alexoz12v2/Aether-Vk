using System;
using Avalonia;
using Avalonia.Controls;
using Avalonia.VisualTree;
using AetherVk.Logic.Services;

namespace AetherVk.UI;

/// <summary>
/// Keeps an <see cref="OverlayWindow"/> pixel-perfectly aligned with a
/// <see cref="NativeControlHost"/> (or any <see cref="Control"/>) by subscribing to
/// position and bounds changes on the host window.
/// <para>
/// Call <see cref="Dispose"/> to close the overlay and unsubscribe all handlers.
/// The router must have <see cref="Logic.Services.IWindowInputRouter.DetachFromWindow"/>
/// called on the overlay <em>before</em> <see cref="Dispose"/> so that the router does
/// not hold a reference to a closed <see cref="TopLevel"/>.
/// </para>
/// </summary>
public sealed class OverlaySynchronizer : IDisposable
{
  private readonly Window _mainWindow;
  private readonly Window _overlayWindow;
  private readonly Control _nativeHost;
  private readonly IDisposable _boundsSubscription;
  private bool _disposed;

  public OverlaySynchronizer(
    Window mainWindow,
    Window overlayWindow,
    Control nativeHost,
    IPlatformWindowService platformWindowService)
  {
    _mainWindow    = mainWindow;
    _overlayWindow = overlayWindow;
    _nativeHost    = nativeHost;

    // 1. Initial show — makes the XID / HWND available.
    //    Passing the owner ensures correct Z-order and that the overlay is minimised/restored
    //    together with the main window.
    _overlayWindow.Show(_mainWindow);

    // 2. Apply EWMH hints that tell the WM this is a dock-type panel (cannot be moved by the
    //    user via Super+LMB or similar WM gestures). All X11 PInvoke stays inside ui-logic.
    //    TryGetPlatformHandle() is a public Avalonia API — not OS-specific PInvoke.
    var xid = _overlayWindow.TryGetPlatformHandle()?.Handle ?? 0;
    platformWindowService.SetWindowNonMoveable(xid);

    // 3. Pulse Hide → Show so the WM re-reads the window type we just set.
    //    The overlay is transparent, so the brief unmap is invisible.
    _overlayWindow.Hide();
    _overlayWindow.Show(_mainWindow);

    // 4. Subscribe AFTER the remap so spurious PositionChanged from the unmap/remap cycle
    //    don't race with initial placement.
    _mainWindow.PositionChanged    += OnPositionChanged;
    _overlayWindow.PositionChanged += OnOverlayPositionChanged;
    _boundsSubscription = _nativeHost
      .GetObservable(Visual.BoundsProperty)
      .Subscribe(_ => SyncBounds());

    SyncBounds(); // initial placement
  }

  private void OnPositionChanged(object? sender, Avalonia.Controls.PixelPointEventArgs e) => SyncBounds();

  private void OnOverlayPositionChanged(object? sender, Avalonia.Controls.PixelPointEventArgs e) => SyncBounds();

  private void SyncBounds()
  {
    if (_nativeHost.GetVisualRoot() is null || _nativeHost.Bounds.Width == 0)
      return;

    // PointToScreen converts local (0,0) of the NativeControlHost to physical screen pixel
    // coordinates, correctly accounting for DPI scaling and parent transforms.
    var origin = _nativeHost.PointToScreen(new Point(0, 0));
    _overlayWindow.Position = origin;
    _overlayWindow.Width    = _nativeHost.Bounds.Width;
    _overlayWindow.Height   = _nativeHost.Bounds.Height;
  }

  public void Dispose()
  {
    if (_disposed) return;
    _disposed = true;
    _mainWindow.PositionChanged    -= OnPositionChanged;
    _overlayWindow.PositionChanged -= OnOverlayPositionChanged;
    _boundsSubscription.Dispose();
    _overlayWindow.Close();
  }
}
