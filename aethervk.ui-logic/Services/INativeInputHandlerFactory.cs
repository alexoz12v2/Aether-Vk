using System;

namespace AetherVk.Logic.Services;

public interface INativeInputHandlerFactory
{
  INativeInputHandler Create(IntPtr handle, string descriptor, TraceLevel traceLevel);
}
