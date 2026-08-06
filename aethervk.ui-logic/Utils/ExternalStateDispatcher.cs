
using System;
using System.Collections.Generic;
using AetherVk.Logic.Services;


/// <summary>
/// Utility class to be instantiated inside the `INativeRuntimeService` implementation.
/// Handles routing native callbacks to C# subscribers for external state handling
/// `internal` as it is supposed to be a member of the native service implementation
/// </summary>
internal class ExternalStateDispatcher
{
  private readonly object _lock = new();

  private readonly Dictionary<ExternalStateType, List<Action<IntPtr>>> _permanentHandlers = [];
  private readonly Dictionary<ExternalStateType, List<Func<IntPtr, bool>>> _transientHandlers = [];

  public void Subscribe(ExternalStateType type, Action<IntPtr> handler)
  {
    lock (_lock)
    {
      if (!_permanentHandlers.TryGetValue(type, out var list))
      {
        list = [];
        _permanentHandlers[type] = list;
      }
      list.Add(handler);
    }
  }

  public void Unsubcribe(ExternalStateType type, Action<IntPtr> handler)
  {
    lock (_lock)
    {
      if (_permanentHandlers.TryGetValue(type, out var list))
      {
        list.Remove(handler);
      }
    }
  }

  public void RegisterTransient(ExternalStateType type, Func<IntPtr, bool> handler)
  {
    lock (_lock)
    {
      if (!_transientHandlers.TryGetValue(type, out var list))
      {
        list = [];
        _transientHandlers[type] = list;
      }
      list.Add(handler);
    }
  }

  /// <summary>
  /// This is the method the master DLL callback will invoke, wire it up in the
  /// `INativeRuntimeService` implementation`
  /// </summary>
  public void OnNativeCallbackReceived(uint stateId, IntPtr data)
  {
    var type = (ExternalStateType)stateId;

    List<Action<IntPtr>> permToInvoke = [];
    List<Func<IntPtr, bool>> transToInvoke = [];

    // Safely copy under lock
    lock (_lock)
    {
      if (_permanentHandlers.TryGetValue(type, out var perm))
        permToInvoke.AddRange(perm);
      if (_transientHandlers.TryGetValue(type, out var trans))
        transToInvoke.AddRange(trans);
    }

    // Invoke permanent handlers
    foreach (var handler in permToInvoke)
    {
      handler(data);
    }

    // Invoke transient handlers and track which ones need removal
    List<Func<IntPtr, bool>> handlerToRemove = [];
    foreach (var handler in transToInvoke)
    {
      bool consumed = handler(data);
      if (consumed)
      {
        handlerToRemove.Add(handler);
      }
    }

    // clean up consumed transient handlers
    if (handlerToRemove.Count > 0)
    {
      lock (_lock)
      {
        if (_transientHandlers.TryGetValue(type, out var trans))
        {
          foreach (var h in handlerToRemove)
            trans.Remove(h);
        }
      }
    }
  }
}
