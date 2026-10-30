using System;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Platform;
using Avalonia.VisualTree;

namespace AetherVk.Controls;

/// <summary>
/// A <see cref="NativeControlHost"/> that hosts the Vulkan rendering surface.
/// This class is intentionally thin: it creates the OS native window handle and
/// delegates all handler lifecycle (input, Vulkan initialisation) to
/// <see cref="Logic.ViewModels.VulkanViewportControlViewModel"/> via
/// <see cref="Logic.ViewModels.VulkanViewportControlViewModel.InitializeHandle"/>.
/// No service locator — all dependencies flow through the ViewModel's DI constructor.
/// </summary>
public class VulkanViewportControl : NativeControlHost
{
  protected override IPlatformHandle CreateNativeControlCore(IPlatformHandle parent)
  {
    // X11 only — Wayland is not supported (no raw input hooks available)
    if (
      System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
        System.Runtime.InteropServices.OSPlatform.Linux
      )
      && parent.HandleDescriptor != "XID"
    )
      throw new NotSupportedException(
        "Wayland is not supported. Launch with AVALONIA_SCREEN_SCALE_FACTORS or force X11."
      );

    var handle = base.CreateNativeControlCore(parent);

    if (
      handle != null
      && handle.Handle != IntPtr.Zero
      && DataContext is Logic.ViewModels.VulkanViewportControlViewModel vm
    )
    {
      // Pass both child handle (XID / HWND) and parent handle (Display* / HINSTANCE)
      // so the ViewModel can build the correct CNativeWindowHandle for AddViewport.
      if (!vm.InitializeHandle(handle.Handle, handle.HandleDescriptor ?? "window", parent.Handle))
        vm.ReportFatalError("Failed to initialize native input handler.");
    }

    return handle!;
  }

  protected override void DestroyNativeControlCore(IPlatformHandle control)
  {
    // The ViewModel owns the input handler and Rx subscriptions — dispose it here
    // so the X11 event loop background thread exits cleanly before the process shuts down.
    if (DataContext is Logic.ViewModels.VulkanViewportControlViewModel vm)
      vm.Dispose();
    base.DestroyNativeControlCore(control);
  }

  protected override void OnSizeChanged(Avalonia.Controls.SizeChangedEventArgs e)
  {
    base.OnSizeChanged(e);
    PushSize();
  }

  private TopLevel? _topLevel;

  protected override void OnAttachedToVisualTree(Avalonia.VisualTreeAttachmentEventArgs e)
  {
    base.OnAttachedToVisualTree(e);
    _topLevel = TopLevel.GetTopLevel(this);
    if (_topLevel is not null)
      _topLevel.ScalingChanged += OnScalingChanged;
    if (_topLevel is Window w)
      w.PositionChanged += OnWindowMoved;
    LayoutUpdated += OnLayoutUpdated;
    PushSize();
  }

  protected override void OnDetachedFromVisualTree(Avalonia.VisualTreeAttachmentEventArgs e)
  {
    if (_topLevel is not null)
      _topLevel.ScalingChanged -= OnScalingChanged;
    if (_topLevel is Window w)
      w.PositionChanged -= OnWindowMoved;
    LayoutUpdated -= OnLayoutUpdated;
    _topLevel = null;
    _origin = null;
    base.OnDetachedFromVisualTree(e);
  }

  // a move to a monitor with another scaling changes the pixels, not the DIP size
  private void OnScalingChanged(object? sender, EventArgs e) => PushSize();

  // the size in physical pixels (DIPs × render scaling), as the native child window and the swapchain
  private void PushSize()
  {
    if (DataContext is Logic.ViewModels.VulkanViewportControlViewModel vm && Bounds.Width > 0 && Bounds.Height > 0)
    {
      vm.NotifyResized(Bounds.Width, Bounds.Height, _topLevel?.RenderScaling ?? TopLevel.GetTopLevel(this)?.RenderScaling ?? 1.0);
      PushOrigin();
    }
  }

  private Avalonia.PixelPoint? _origin;
  private double _originScaling;

  // the desktop position (physical px) the billboards use: window moves and layout changes move
  // it, a monitor of another scaling changes the px per DIP
  private void PushOrigin()
  {
    if (DataContext is not Logic.ViewModels.VulkanViewportControlViewModel vm || this.GetVisualRoot() is null)
      return;
    var origin = this.PointToScreen(new Avalonia.Point(0, 0));
    double scaling = _topLevel?.RenderScaling ?? TopLevel.GetTopLevel(this)?.RenderScaling ?? 1.0;
    if (_origin == origin && _originScaling == scaling)
      return;
    _origin = origin;
    _originScaling = scaling;
    vm.NotifyScreenOrigin(origin.X, origin.Y, scaling);
  }

  private void OnWindowMoved(object? sender, Avalonia.Controls.PixelPointEventArgs e) => PushOrigin();

  private void OnLayoutUpdated(object? sender, EventArgs e) => PushOrigin();
}
