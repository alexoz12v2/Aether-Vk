using System;

namespace AetherVk.Logic.Services;

// handles both X11 and Wayland

#if !TARGET_IS_LINUX

public unsafe class LinuxNativeInputHandler(IntPtr handle, string handleDescriptor, TraceLevel traceLevel, IUiThreadDispatcher dispatcher, ISchedulerProvider schedulerProvider)
  : NativeInputHandlerBase(handle, handleDescriptor, traceLevel, dispatcher, schedulerProvider)
{
  protected override bool HookEvents()
  {
    return false;
  }

  protected override void UnhookEvents()
  {
  }

  protected override void DoSetSolidColor(byte r, byte g, byte b)
  {
  }
}

#else

public unsafe class LinuxNativeInputHandler(IntPtr handle, string handleDescriptor, TraceLevel traceLevel, IUiThreadDispatcher dispatcher, ISchedulerProvider schedulerProvider)
  : NativeInputHandlerBase(handle, handleDescriptor, traceLevel, dispatcher, schedulerProvider)
{
  protected override bool HookEvents()
  {
    return false;
  }

  protected override void UnhookEvents()
  {
  }

  protected override void DoSetSolidColor(byte r, byte g, byte b)
  {
  }
}

#endif
