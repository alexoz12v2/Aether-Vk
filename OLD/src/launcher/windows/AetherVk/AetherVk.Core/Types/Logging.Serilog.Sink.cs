using Microsoft.Extensions.Logging;
using Serilog.Core;
using Serilog.Events;

namespace AetherVk.Core.Types
{
    public sealed class UILogSink(ILogEventHub hub) : ILogEventSink
    {
        private readonly ILogEventHub _Hub = hub;

        public void Emit(LogEvent logEvent)
        {
            LogEventEntry entry = new(
                logEvent.Timestamp,
                ToLevel(logEvent.Level),
                logEvent.RenderMessage(),
                logEvent.Exception);
            _Hub.Publish(entry);
        }

        // log level from serilog to microsoft extension log level
        private static LogLevel ToLevel(LogEventLevel level)
        {
            return level switch
            {
                LogEventLevel.Verbose => LogLevel.Trace,
                LogEventLevel.Debug => LogLevel.Debug,
                LogEventLevel.Information => LogLevel.Information,
                LogEventLevel.Warning => LogLevel.Warning,
                LogEventLevel.Error => LogLevel.Error,
                LogEventLevel.Fatal => LogLevel.Critical,
                _ => LogLevel.Information
            };
        }
    }
}