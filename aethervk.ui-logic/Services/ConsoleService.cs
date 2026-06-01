using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using AetherVk.Logic.Models;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.Services;

public class ConsoleService : ObservableObject, IDisposable
{
  // UI-agnostic bulk update notification collection
  public ObservableRangeCollection<string> Messages { get; } = [];

  private readonly ConcurrentQueue<string> _messageQueue = new();
  private readonly Timer _flushTimer;

  // Bound the UI log history so Avalonia doesn't bloat memory
  private const int MaxLogLines = 1500;

  // Prevent overlapping timer executions if the thread pool is busy
  private int _isFlushing;

  private readonly IUiThreadDispatcher? _uiThreadDispatcher;

  public ConsoleService(IUiThreadDispatcher? uiThreadDispatcher = null)
  {
    _uiThreadDispatcher = uiThreadDispatcher;
    _flushTimer = new Timer(FlushMessages, null, 100, 100);
  }

  // Do not call any synchronous functions like `Console.WriteLine` here!
  public void Log(string message)
  {
    var msg = $"[{DateTime.Now:HH:mm:ss}] {message.TrimEnd('\n', '\r')}";
    Console.WriteLine(msg);
    _messageQueue.Enqueue(msg);
  }

  private void FlushMessages(object? state)
  {
    // if we are already flushing, return immediately
    if (Interlocked.CompareExchange(ref _isFlushing, 1, 0) != 0)
      return;
    try
    {
      if (_messageQueue.IsEmpty)
        return;

      // Completely darin the queue. If we limit the amount we drain, we'll fill in the queue and
      // run out of memory
      var batch = new List<string>();
      while (_messageQueue.TryDequeue(out var msg))
      {
        batch.Add(msg);
      }

      if (batch.Count == 0)
        return;

      // Do slow I/O printing safely on background
#if DEBUG
      foreach (var m in batch)
      {
        System.Diagnostics.Debug.WriteLine(m);
      }
#endif
      // Pre-trim the batch in case of massive log spike exceeding max log lines
      if (batch.Count > MaxLogLines)
      {
        batch.RemoveRange(0, batch.Count - MaxLogLines);
      }

      if (_uiThreadDispatcher != null)
        _uiThreadDispatcher.Dispatch(UpdateUi);
      else
        UpdateUi();

      void UpdateUi()
      {
        // Bulk insert. Triggers exactly 1 UI update
        Messages.AddRange(batch);
        // Trim the oldest messages dynamically to maintain Memory cap
        var excess = Messages.Count - MaxLogLines;
        if (excess > 0)
        {
          Messages.RemoveRange(0, excess);
        }
      }
    }
    finally
    {
      Interlocked.Exchange(ref _isFlushing, 0);
    }
  }

  public void Clear()
  {
#if NETSTANDARD2_0
    // Fallback path for .NET Standard 2.0 / .NET Framework
    while (_messageQueue.TryDequeue(out _)) { }
#else
    // Fast path for modern .NET
    _messageQueue.Clear();
#endif

    if (_uiThreadDispatcher != null)
    {
      _uiThreadDispatcher.Dispatch(ClearUi);
    }
    else
    {
      ClearUi();
    }

    return;

    void ClearUi() => Messages.Clear();
  }

  public void Dispose()
  {
    _flushTimer.Dispose();
  }
}
