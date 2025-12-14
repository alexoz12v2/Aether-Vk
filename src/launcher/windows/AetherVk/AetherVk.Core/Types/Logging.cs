using Microsoft.Extensions.Logging;

namespace AetherVk.Core.Types
{
    public sealed record LogEventEntry(
        DateTimeOffset Timestamp,
        LogLevel Level,
        string Message,
        Exception? Exception);

    public interface ILogEventHub
    {
        void Publish(LogEventEntry entry);
        IDisposable Subscribe(Action<LogEventEntry> handler);
    }

    public sealed class LogEventHub : ILogEventHub
    {
        private readonly Lock _Lock = new();
        private readonly List<Action<LogEventEntry>> _Subscribers = [];

        public void Publish(LogEventEntry entry)
        {
            Action<LogEventEntry>[] subs;
            lock (_Lock)
            {
                subs = [.. _Subscribers];
            }
            foreach (Action<LogEventEntry> sub in subs)
            {
                sub(entry);
            }
        }

        private sealed class Subscription(Action unsubscribe) : IDisposable
        {
            public void Dispose()
            {
                unsubscribe();
            }
        }

        public IDisposable Subscribe(Action<LogEventEntry> handler)
        {
            lock (_Lock)
            {
                _Subscribers.Add(handler);
            }
            return new Subscription(() =>
            {
                lock (_Lock)
                {
                    _ = _Subscribers.Remove(handler);
                }
            });
        }
    }
}