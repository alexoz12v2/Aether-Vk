using System;
using System.IO;
using System.Threading;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests;

public class StartupWatchdogTests : IDisposable
{
  private readonly string _directory = Path.Combine(
    Path.GetTempPath(),
    "aethervk-watchdog-tests",
    Guid.NewGuid().ToString("N")
  );

  public void Dispose()
  {
    try
    {
      Directory.Delete(_directory, recursive: true);
    }
    catch (Exception) { }
  }

  [Fact]
  public void SilentLog_WritesLastLineAndDumpThenStops()
  {
    using var sink = new LogFileSink(_directory);
    sink.Write("[Shader] creating module \"dust_splat.comp\" (512 words)");

    string? dumpPath = null;
    var dumped = new ManualResetEventSlim(false);
    var watchdog = new StartupWatchdog(
      sink,
      TimeSpan.FromMilliseconds(200),
      path =>
      {
        dumpPath = path;
        dumped.Set();
        return true;
      },
      TimeSpan.FromMilliseconds(20)
    );
    watchdog.Start();

    Assert.True(dumped.Wait(TimeSpan.FromSeconds(5)));
    Assert.True(watchdog.Fired);
    Assert.Contains("aethervk-hang-", dumpPath);

    // The log names the last step, and says where the dump went. The dump is written before the
    // final log line, so wait for that line rather than reading straight after the callback.
    var text = WaitForText(sink.FilePath, "Dump written");
    Assert.Contains("[WATCHDOG] No log output", text);
    Assert.Contains("dust_splat.comp", text);
  }

  [Fact]
  public void ActiveLog_DoesNotFire_AndDisarmStopsIt()
  {
    using var sink = new LogFileSink(_directory);
    var fired = false;
    var watchdog = new StartupWatchdog(
      sink,
      TimeSpan.FromMilliseconds(300),
      _ =>
      {
        fired = true;
        return true;
      },
      TimeSpan.FromMilliseconds(20)
    );
    watchdog.Start();

    // Lines every 50 ms keep the log busy for longer than the timeout
    for (var i = 0; i < 20; i++)
    {
      sink.Write($"progress {i}");
      Thread.Sleep(50);
    }
    Assert.False(watchdog.Fired);

    watchdog.Disarm();
    Thread.Sleep(500);
    Assert.False(fired);
  }

  private static string WaitForText(string path, string expected)
  {
    var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(5);
    while (true)
    {
      var text = ReadWithSharedAccess(path);
      if (text.Contains(expected) || DateTime.UtcNow > deadline)
        return text;

      Thread.Sleep(20);
    }
  }

  private static string ReadWithSharedAccess(string path)
  {
    using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
    using var reader = new StreamReader(stream);
    return reader.ReadToEnd();
  }
}
