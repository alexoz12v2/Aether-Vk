using AetherVk.Logic.Services;
using System.Reactive.Concurrency;
using Avalonia.ReactiveUI;

namespace AetherVk.Services;

public class AvaloniaSchedulerProvider : ISchedulerProvider
{
  // links to Avalonia UI Thread
  public IScheduler MainThread => AvaloniaScheduler.Instance;

  // Standard .NET background thread pool
  public IScheduler Background => TaskPoolScheduler.Default;
}
