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
    var handle = base.CreateNativeControlCore(parent);

    bool success = false;
    if (handle != null && handle.Handle != IntPtr.Zero)
    {
      var factory = App.Host?.Services.GetRequiredService<INativeInputHandlerFactory>();
      if (factory != null)
      {
        _nativeInputHandler = factory.Create(handle.Handle, TraceLevel.Max);
        success = true;
      }
    }

    if (!success)
    {
      if (DataContext is AetherVk.Logic.ViewModels.VulkanViewportControlViewModel vm)
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
