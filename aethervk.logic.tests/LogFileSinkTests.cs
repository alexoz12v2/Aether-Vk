using System;
using System.IO;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests;

public class LogFileSinkTests : IDisposable
{
  private readonly string _directory = Path.Combine(
    Path.GetTempPath(),
    "aethervk-logsink-tests",
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
  public void Flush_WritesQueuedLinesToTheRunFile()
  {
    using var sink = new LogFileSink(_directory);

    sink.Write("first line");
    sink.Write("second line");
    Assert.True(sink.Flush(TimeSpan.FromSeconds(5)));

    var text = ReadWithSharedAccess(sink.FilePath);
    Assert.Contains("first line", text);
    Assert.Contains("second line", text);
  }

  [Fact]
  public void Constructor_KeepsOnlyTheNewestRunFiles()
  {
    Directory.CreateDirectory(_directory);
    for (var i = 0; i < 25; i++)
    {
      // Timestamp-style names, so the lexical order is the run order
      File.WriteAllText(Path.Combine(_directory, $"aethervk-2000010{i:D2}-000000-1.log"), "old");
    }

    using var sink = new LogFileSink(_directory);
    sink.Flush(TimeSpan.FromSeconds(5));

    // 19 old runs plus the one just started
    Assert.Equal(20, Directory.GetFiles(_directory, "aethervk-*.log").Length);
  }

  [Fact]
  public void Write_AfterDispose_DoesNotThrow()
  {
    var sink = new LogFileSink(_directory);
    sink.Dispose();

    sink.Write("late line");
    Assert.False(sink.Flush(TimeSpan.FromMilliseconds(50)));
  }

  private static string ReadWithSharedAccess(string path)
  {
    using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
    using var reader = new StreamReader(stream);
    return reader.ReadToEnd();
  }
}
