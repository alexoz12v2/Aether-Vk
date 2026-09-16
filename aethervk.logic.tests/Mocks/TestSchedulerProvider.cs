using System.Reactive.Concurrency;
using AetherVk.Logic.Services;
using Microsoft.Reactive.Testing;

namespace AetherVk.Logic.Tests.Mocks;

public class TestSchedulerProvider : ISchedulerProvider
{
    public TestScheduler MainThread { get; } = new TestScheduler();
    public TestScheduler Background { get; } = new TestScheduler();

    IScheduler ISchedulerProvider.MainThread => MainThread;
    IScheduler ISchedulerProvider.Background => Background;
}
