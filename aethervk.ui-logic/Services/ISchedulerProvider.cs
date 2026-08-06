
using System.Reactive.Concurrency;

namespace AetherVk.Logic.Services;

/// <summary>
/// Interface to provide `System.Reactive` schedulers independently from which UI library in use
/// </summary>
public interface ISchedulerProvider
{
  IScheduler MainThread { get; }
  IScheduler Background { get; }
}
