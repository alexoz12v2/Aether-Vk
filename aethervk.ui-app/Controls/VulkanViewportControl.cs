using Avalonia.Controls;
using Avalonia.Platform;
using System;
using AetherVk.Logic.Services;
using Microsoft.Extensions.DependencyInjection;

namespace AetherVk.Controls;

public class VulkanViewportControl : NativeControlHost
{
  private INativeInputHandler? _nativeInputHandler;

  protected override IPlatformHandle CreateNativeControlCore(IPlatformHandle parent)
  {
    // Check: We don't want wayland (TODO fatal error message)
    if (System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Linux) &&
        parent.HandleDescriptor != "XID")
      throw new NotSupportedException("We don't support wayland. How did it even get enabled?");

    var handle = base.CreateNativeControlCore(parent);

    bool success = false;
    if (handle != null && handle.Handle != IntPtr.Zero)
    {
      var factory = App.Host?.Services.GetRequiredService<INativeInputHandlerFactory>();
      if (factory != null)
      {
        _nativeInputHandler = factory.Create(handle.Handle, handle.HandleDescriptor ?? "window", TraceLevel.Max);
        success = true;
      }
    }

    if (!success)
    {
      if (DataContext is Logic.ViewModels.VulkanViewportControlViewModel vm)
      {
        vm.ReportFatalError("Failed to initialize OS Window Handle or resolve INativeInputHandlerFactory.");
      }
    }

    return handle!;
  }

  protected override void DestroyNativeControlCore(IPlatformHandle control)
  {
    _nativeInputHandler?.Dispose();
    _nativeInputHandler = null;
    base.DestroyNativeControlCore(control);
  }
}
