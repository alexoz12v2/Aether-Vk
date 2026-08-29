using System;
using System.Runtime.InteropServices;
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
/// On Linux/XWayland: after <c>Show()</c> allocates the XID, <c>override_redirect</c> is set
/// via <c>XChangeWindowAttributes</c> and the window is remapped. This instructs Xwayland to
/// export the overlay as an independent Wayland surface so Mutter can GPU-composite it with
/// correct ARGB alpha-blending AFTER the Vulkan sub-surface — bypassing both the top-level
/// Z-order problem and the child-window backing-store transparency hole.
/// Because override_redirect windows are unmanaged by the WM, this class manually hides/shows
/// the overlay in response to MainWindow minimize and focus-loss events.
/// </para>
/// <para>
/// On Windows and macOS the owned/child-window relationship established by
/// <c>Show(ownerWindow)</c> provides the correct Z-order automatically, so the
/// <see cref="IPlatformWindowService.SetOverlayAbove"/> calls are no-ops on those platforms.
/// </para>
/// <para>
/// Call <see cref="Dispose"/> to close the overlay and unsubscribe all handlers.
/// The router must have <see cref="Logic.Services.IWindowInputRouter.DetachFromWindow"/>
/// called on the overlay <em>before</em> <see cref="Dispose"/> so that the router does
/// not hold a reference to a closed <see cref="TopLevel"/>.
/// </para>
/// </summary>
public sealed class OverlaySynchronizer : IDisposable
{
  private readonly Window                  _mainWindow;
  private readonly Window                  _overlayWindow;
  private readonly Control                 _nativeHost;
  private readonly IPlatformWindowService  _platformWindowService;
  private readonly IDisposable             _boundsSubscription;
  private bool _disposed;

  // Cached OS handles — retrieved once after Show() so we can pass them to
  // the platform service without re-querying TryGetPlatformHandle on every event.
  private readonly nint _overlayHandle;
  private nint          _rootHandle;   // X11 root window; 0 on Windows/macOS (ignored)

  // True on Linux when override_redirect has been applied and remapped.
  // Drives the manual hide/show on focus and minimize events.
  private readonly bool _isOverrideRedirect;

  // Last bounds we *requested* (not what the WM reported back).
  // Guards against the X11 async PositionChanged ping-pong loop on non-override_redirect paths.
  private Avalonia.PixelPoint? _lastRequestedPos;
  private Avalonia.Size?       _lastRequestedSize;

  public OverlaySynchronizer(
    Window mainWindow,
    Window overlayWindow,
    Control nativeHost,
    IPlatformWindowService platformWindowService)
  {
    _mainWindow            = mainWindow;
    _overlayWindow         = overlayWindow;
    _nativeHost            = nativeHost;
    _platformWindowService = platformWindowService;

    // 1. Initial show — makes the XID / HWND / NSWindow available and establishes the
    //    ownership relationship (XSetTransientForHint on X11, SetWindowLongPtr GWL_HWNDPARENT
    //    on Windows, addChildWindow:ordered:above on macOS).
    _overlayWindow.Show(_mainWindow);

    // 2. Retrieve the overlay's platform handle now that the window is mapped.
    _overlayHandle = _overlayWindow.TryGetPlatformHandle()?.Handle ?? 0;

    if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
    {
      // 3. Linux/XWayland: set override_redirect on the XID.
      //    This tells Xwayland to stop asking Mutter to manage this window and instead
      //    export it as its own independent Wayland surface. Mutter then GPU-composites
      //    it with full ARGB alpha-blending AFTER the Vulkan sub-surface.
      _platformWindowService.SetOverlayOverrideRedirect(_overlayHandle);

      // 4. Pulse Hide → Show so Xwayland re-exports the window with override_redirect active.
      //    The overlay is transparent, so the brief unmap is invisible.
      _overlayWindow.Hide();
      _overlayWindow.Show(_mainWindow);
      _isOverrideRedirect = true;

      // 5. Manual WM responsibilities: override_redirect windows are completely ignored by
      //    Mutter, so we must manually hide/show in response to window state changes.
      _mainWindow.Activated          += OnMainWindowActivated;
      _mainWindow.Deactivated        += OnMainWindowDeactivated;
      _mainWindow.PropertyChanged    += OnMainWindowPropertyChanged;
    }
    else
    {
      // 3. Non-Linux: apply EWMH UTILITY type + empty ALLOWED_ACTIONS, then pulse
      //    Hide → Show so the WM re-reads them. No-op on Windows/macOS.
      _platformWindowService.SetWindowAsCompanionPanel(_overlayHandle);
      _overlayWindow.Hide();
      _overlayWindow.Show(_mainWindow);

      // Raise via _NET_WM_STATE_ABOVE (no-op on Windows/macOS).
      _rootHandle = _platformWindowService.GetRootWindowHandle();
      _platformWindowService.SetOverlayAbove(_overlayHandle, _rootHandle, raise: true);

      // Focus-driven Z-order toggle (no-op on Windows/macOS).
      _mainWindow.Activated   += OnMainWindowActivated;
      _mainWindow.Deactivated += OnMainWindowDeactivated;
    }

    // 6. Subscribe to position/bounds AFTER setup to avoid spurious events from the remap cycle.
    _mainWindow.PositionChanged    += OnPositionChanged;
    _overlayWindow.PositionChanged += OnOverlayPositionChanged;

    _boundsSubscription = _nativeHost
      .GetObservable(Visual.BoundsProperty)
      .Subscribe(_ => SyncBounds());

    SyncBounds(); // initial placement
  }

  private void OnPositionChanged(object? sender, Avalonia.Controls.PixelPointEventArgs e)
    => SyncBounds();

  private void OnOverlayPositionChanged(object? sender, Avalonia.Controls.PixelPointEventArgs e)
    => SyncBounds();

  private void OnMainWindowActivated(object? sender, EventArgs e)
  {
    if (_isOverrideRedirect)
    {
      // override_redirect: WM won't show it for us — do it manually.
      if (_mainWindow.WindowState != WindowState.Minimized)
        _overlayWindow.Show(_mainWindow);
    }
    else
    {
      _platformWindowService.SetOverlayAbove(_overlayHandle, _rootHandle, raise: true);
    }
  }

  private void OnMainWindowDeactivated(object? sender, EventArgs e)
  {
    if (_isOverrideRedirect)
    {
      // override_redirect: hide when the app loses focus so the overlay doesn't float
      // above foreground windows of other applications within Xwayland's Z-stack.
      _overlayWindow.Hide();
    }
    else
    {
      _platformWindowService.SetOverlayAbove(_overlayHandle, _rootHandle, raise: false);
    }
  }

  private void OnMainWindowPropertyChanged(object? sender, Avalonia.AvaloniaPropertyChangedEventArgs e)
  {
    if (e.Property != Window.WindowStateProperty) return;

    // override_redirect: WM won't minimize/restore the overlay alongside the main window.
    var state = (WindowState)(e.NewValue ?? WindowState.Normal);
    if (state == WindowState.Minimized)
      _overlayWindow.Hide();
    else if (_mainWindow.IsActive)
      _overlayWindow.Show(_mainWindow);
  }

  private void SyncBounds()
  {
    if (_nativeHost.GetVisualRoot() is null || _nativeHost.Bounds.Width == 0)
      return;

    // PointToScreen converts local (0,0) of the NativeControlHost to physical screen pixel
    // coordinates, correctly accounting for DPI scaling and parent transforms.
    // Works for both override_redirect top-levels (Linux) and owned top-levels (Win/macOS).
    var origin = _nativeHost.PointToScreen(new Point(0, 0));
    var size   = _nativeHost.Bounds.Size;

    // Short-circuit if we already requested these exact bounds.
    // Breaks the async X11 WM PositionChanged ping-pong on non-override_redirect paths.
    if (_lastRequestedPos == origin && _lastRequestedSize == size)
      return;

    _lastRequestedPos  = origin;
    _lastRequestedSize = size;

    _overlayWindow.Position = origin;
    _overlayWindow.Width    = size.Width;
    _overlayWindow.Height   = size.Height;
  }

  public void Dispose()
  {
    if (_disposed) return;
    _disposed = true;
    _mainWindow.PositionChanged    -= OnPositionChanged;
    _overlayWindow.PositionChanged -= OnOverlayPositionChanged;
    _mainWindow.Activated          -= OnMainWindowActivated;
    _mainWindow.Deactivated        -= OnMainWindowDeactivated;
    if (_isOverrideRedirect)
      _mainWindow.PropertyChanged  -= OnMainWindowPropertyChanged;
    _boundsSubscription.Dispose();
    _overlayWindow.Close();
  }
}

