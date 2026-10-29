using System;
using System.IO;
using System.Linq;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The crash report next to the run logs (reviewer report 2026-10-29: a crash after a few minutes
/// came with nothing to read): the cause, the version and OS, and the last 200 log lines, for an
/// unhandled exception and for a native engine panic line.
/// </summary>
public sealed class CrashReportTests : IDisposable
{
  private readonly string _dir = Path.Combine(Path.GetTempPath(), "avk_crash_" + Guid.NewGuid().ToString("N"));

  public void Dispose()
  {
    try
    {
      Directory.Delete(_dir, recursive: true);
    }
    catch (Exception) { }
  }

  private static string ReadShared(string path)
  {
    using var s = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
    return new StreamReader(s).ReadToEnd();
  }

  [Fact]
  public void AReport_HoldsTheCauseAndTheLast200Lines()
  {
    using var sink = new LogFileSink(_dir);
    for (int i = 0; i < 250; i++)
      sink.Write($"line {i}");

    var path = CrashReport.Write(sink, "Unhandled exception", "System.InvalidOperationException: boom");

    Assert.NotNull(path);
    Assert.StartsWith(Path.Combine(_dir, "aethervk-crash-"), path);
    var text = File.ReadAllText(path!);
    Assert.Contains("Aether-Vk crash report: Unhandled exception", text);
    Assert.Contains("System.InvalidOperationException: boom", text);
    Assert.Contains("Run log: " + sink.FilePath, text);
    Assert.DoesNotContain("line 49\n", text.Replace("\r\n", "\n"));
    Assert.Contains("line 50", text);
    Assert.Contains("line 249", text);
    Assert.Equal(LogFileSink.RecentLineCount, text.Split('\n').Count(l => l.StartsWith("line ")));
    // the run log names the report, and is on disk already (flushed)
    Assert.Contains("crash report written to " + path, ReadShared(sink.FilePath));
  }

  [Fact]
  public void ANativePanicLine_WritesAReportFromTheLoggingThread()
  {
    using var sink = new LogFileSink(_dir);
    var console = new ConsoleService(fileSink: sink);
    console.Log("[INFO] before");

    console.Log("[PANIC] thread 'logic' panicked at rlib/src/scene/dust.rs:1:1:\nboom\nbacktrace:\n   0: frame");

    var report = Directory.GetFiles(_dir, "aethervk-crash-*.txt").Single();
    var text = File.ReadAllText(report);
    Assert.Contains("Aether-Vk crash report: Native engine panic", text);
    Assert.Contains("panicked at rlib/src/scene/dust.rs", text);
    Assert.Contains("[INFO] before", text);
  }

  [Fact]
  public void OnlyTheNewestReportsAreKept()
  {
    using var sink = new LogFileSink(_dir);
    var start = new DateTime(2026, 10, 29, 12, 0, 0);
    for (int i = 0; i < 13; i++)
      Assert.NotNull(CrashReport.Write(sink, "test", "x", start.AddSeconds(i)));

    var kept = Directory.GetFiles(_dir, "aethervk-crash-*.txt").Select(Path.GetFileName).OrderBy(f => f).ToArray();
    Assert.Equal(CrashReport.KeptReports, kept.Length);
    Assert.StartsWith("aethervk-crash-20261029-120003-", kept[0]);
  }

  [Fact]
  public void AnUnwritableFolder_ReturnsNull_AndDoesNotThrow()
  {
    Directory.CreateDirectory(_dir);
    var blocker = Path.Combine(_dir, "not-a-dir");
    File.WriteAllText(blocker, "file");
    using var sink = new LogFileSink(Path.Combine(blocker, "logs"));

    Assert.Null(CrashReport.Write(sink, "test", "x"));
  }
}
