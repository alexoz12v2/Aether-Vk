using System;
using System.Runtime.InteropServices;

namespace AetherVk.Logic.Services;

public class NativeInputHandlerFactory(IUiThreadDispatcher dispatcher, ISchedulerProvider schedulerProvider) : INativeInputHandlerFactory
{
  private readonly IUiThreadDispatcher _dispatcher = dispatcher;
  private readonly ISchedulerProvider _schedulerProvider = schedulerProvider;

  public INativeInputHandler Create(IntPtr handle, string descriptor, TraceLevel traceLevel)
  {
#if IS_JIT_COMPILED || TARGET_IS_WINDOWS
    if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
    {
      return new WindowsNativeInputHandler(handle, descriptor, traceLevel, _dispatcher, _schedulerProvider);
    }
#endif

#if IS_JIT_COMPILED || TARGET_IS_LINUX
    if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
    {
      return new LinuxNativeInputHandler(handle, descriptor, traceLevel, _dispatcher, _schedulerProvider);
    }
#endif

#if IS_JIT_COMPILED || TARGET_IS_OSX
    if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
    {
      return new MacNativeInputHandler(handle, descriptor, traceLevel, _dispatcher, _schedulerProvider);
    }
#endif

    throw new PlatformNotSupportedException("OS platform not supported for native input handling.");
  }
}
