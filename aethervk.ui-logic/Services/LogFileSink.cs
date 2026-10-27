using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace AetherVk.Logic.Services;

/// <summary>
/// Appends every log line to one file per run, so a release build (which has no console on Windows)
/// still leaves a log to send. Lines are queued and written on a background thread; a failure to
/// write is swallowed so logging can never take the app down.
/// </summary>
public sealed class LogFileSink : IDisposable
{
  private const string Prefix = "aethervk";

  // Newest runs kept in the folder, the current run included
  private const int KeptFiles = 20;

  private readonly BlockingCollection<Action<StreamWriter>> _work = new();
  private readonly Thread _writer;

  public string FilePath { get; }

  public LogFileSink(string directory)
  {
    try
    {
      Directory.CreateDirectory(directory);
      RemoveOldFiles(directory);
    }
    catch (Exception)
    {
      // Without the folder the writer below fails silently and the queue is drained
    }

    FilePath = Path.Combine(
      directory,
      $"{Prefix}-{DateTime.Now:yyyyMMdd-HHmmss}-{Process.GetCurrentProcess().Id}.log"
    );

    _writer = new Thread(WriteLoop) { IsBackground = true, Name = "LogFileSink" };
    _writer.Start();
  }

  /// <summary>
  /// Log folder for this platform: <c>%LOCALAPPDATA%\Aether-Vk\logs</c> on Windows,
  /// <c>~/Library/Logs/Aether-Vk</c> on macOS, <c>$XDG_STATE_HOME/Aether-Vk/logs</c> on Linux.
  /// </summary>
  public static LogFileSink CreateDefault() => new LogFileSink(DefaultDirectory());

  public static string DefaultDirectory()
  {
    if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
    {
      return Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Aether-Vk",
        "logs"
      );
    }

    if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
    {
      return Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        "Library",
        "Logs",
        "Aether-Vk"
      );
    }

    var stateHome = Environment.GetEnvironmentVariable("XDG_STATE_HOME");
    if (string.IsNullOrEmpty(stateHome))
    {
      stateHome = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".local",
        "state"
      );
    }

    return Path.Combine(stateHome, "Aether-Vk", "logs");
  }

  /// <summary>Queues one line. Never blocks the caller.</summary>
  public void Write(string line)
  {
    Enqueue(w => w.WriteLine(line));
  }

  /// <summary>
  /// Waits until everything queued so far is on disk, or the timeout passes. Call before the
  /// process exits on a crash, so the last lines are not lost.
  /// </summary>
  public bool Flush(TimeSpan timeout)
  {
    using var done = new ManualResetEventSlim(false);
    if (!Enqueue(w => { w.Flush(); done.Set(); }))
      return false;

    return done.Wait(timeout);
  }

  // False once the sink is disposed: BlockingCollection throws on adds after CompleteAdding
  private bool Enqueue(Action<StreamWriter> work)
  {
    try
    {
      return _work.TryAdd(work);
    }
    catch (InvalidOperationException)
    {
      return false;
    }
  }

  private void WriteLoop()
  {
    StreamWriter? writer = null;
    try
    {
      var stream = new FileStream(FilePath, FileMode.Create, FileAccess.Write, FileShare.Read);
      writer = new StreamWriter(stream, new UTF8Encoding(false));
    }
    catch (Exception)
    {
      // No file: the loop below still drains the queue so writers never block
    }

    foreach (var work in _work.GetConsumingEnumerable())
    {
      if (writer == null)
        continue;

      try
      {
        work(writer);

        // Flush once the queue is empty: the file stays current, and a burst costs one flush
        if (_work.Count == 0)
          writer.Flush();
      }
      catch (Exception)
      {
        // A flush waiter whose timeout passed can see its event disposed; keep writing
      }
    }

    try
    {
      writer?.Dispose();
    }
    catch (Exception) { }
  }

  private static void RemoveOldFiles(string directory)
  {
    // File names start with the run's timestamp, so the newest sort last
    var old = Directory
      .GetFiles(directory, Prefix + "-*.log")
      .OrderByDescending(f => f, StringComparer.Ordinal)
      .Skip(KeptFiles - 1);

    foreach (var file in old)
    {
      try
      {
        File.Delete(file);
      }
      catch (Exception) { }
    }
  }

  public void Dispose()
  {
    _work.CompleteAdding();
    _writer.Join(TimeSpan.FromSeconds(2));
  }
}
