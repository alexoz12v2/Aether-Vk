using System;
using Avalonia.Controls;
using Avalonia.Platform;

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
}
