using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Threading;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.Services;

public partial class ConsoleService : ObservableObject, IDisposable
{
  public ObservableCollection<string> Messages { get; } = new ObservableCollection<string>();

  private readonly ConcurrentQueue<string> _messageQueue = new();
  private readonly Timer _flushTimer;

  public ConsoleService()
  {
    _flushTimer = new Timer(FlushMessages, null, 100, 100);
  }

  public void Log(string message)
  {
    var msg = $"[{DateTime.Now:HH:mm:ss}] {message}";
#if DEBUG
    System.Console.WriteLine(msg);
    System.Diagnostics.Debug.WriteLine(msg);
#endif
    _messageQueue.Enqueue(msg);
  }

  private void FlushMessages(object? state)
  {
    if (_messageQueue.IsEmpty)
      return;

    var batch = new List<string>();
    while (_messageQueue.TryDequeue(out var msg))
    {
      batch.Add(msg);
      if (batch.Count >= 200)
        break; // Limit batch size to prevent UI freeze
    }

    if (batch.Count > 0)
    {
      if (ServiceLocator.DispatchToUI != null)
      {
        ServiceLocator.DispatchToUI(() =>
        {
          foreach (var m in batch)
            Messages.Add(m);
        });
      }
      else
      {
        foreach (var m in batch)
          Messages.Add(m);
      }
    }
  }

  public void Clear()
  {
    while (_messageQueue.TryDequeue(out _)) { }

    if (ServiceLocator.DispatchToUI != null)
    {
      ServiceLocator.DispatchToUI(() => Messages.Clear());
    }
    else
    {
      Messages.Clear();
    }
  }

  public void Dispose()
  {
    _flushTimer?.Dispose();
  }
}
