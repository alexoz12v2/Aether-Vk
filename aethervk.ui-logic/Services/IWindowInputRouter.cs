
using System;

namespace AetherVk.Logic.Services;

public interface IWindowInputRouter : IDisposable
{
  /// <summary>
  /// Attaches the router to a specific window root (eg Avalonia TopLevel or Window).
  /// PAssed as object to keep logic layer UI-agnostic
  /// </summary>
  void AttachToWindow(object windowRoot);
}
