using System;

namespace AetherVk.Logic.Services;

public interface INativeInputHandler : IDisposable
{
  /// <summary>
  /// Requests that the underlying OS window (viewport XID on Linux, HWND on Windows, NSView on
  /// macOS) receives keyboard focus so that native key events are delivered to this handler.
  /// No-op on platforms where focus is managed differently.
  /// </summary>
  void FocusViewportWindow();
}
