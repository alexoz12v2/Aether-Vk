using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;

namespace AetherVk.Logic.Services;

/// <summary>
/// Startup watchdog. While it runs, if the log has been silent for the timeout, it records the last
/// line and writes a minidump of the process (every thread's stack) next to the log, so a hang, such
/// as a driver stalling in shader compilation, leaves a file to send. It fires once and stops.
/// <see cref="Disarm"/> is called when startup completes, so a later idle period is not a hang.
/// </summary>
public sealed class StartupWatchdog
{
  private readonly LogFileSink _sink;
  private readonly TimeSpan _timeout;
  private readonly TimeSpan _poll;
  private readonly Func<string, bool> _writeDump;
  private readonly ManualResetEventSlim _stop = new ManualResetEventSlim(false);
  private readonly Thread _thread;
  private volatile bool _fired;

  /// <param name="writeDump">Writes a dump to the path and reports success. Defaults to a Windows minidump.</param>
  /// <param name="poll">How often the log's silence is checked.</param>
  public StartupWatchdog(
    LogFileSink sink,
    TimeSpan timeout,
    Func<string, bool>? writeDump = null,
    TimeSpan? poll = null
  )
  {
    _sink = sink;
    _timeout = timeout;
    _poll = poll ?? TimeSpan.FromSeconds(1);
    _writeDump = writeDump ?? MiniDump.TryWrite;
    _thread = new Thread(Run) { IsBackground = true, Name = "StartupWatchdog" };
  }

  public bool Fired => _fired;

  public void Start()
  {
    _thread.Start();
  }

  /// <summary>Startup is over: stop watching. Safe to call before <see cref="Start"/> or twice.</summary>
  public void Disarm()
  {
    _stop.Set();
  }

  private void Run()
  {
    while (!_stop.Wait(_poll))
    {
      var silent = DateTime.UtcNow - _sink.LastWriteUtc;
      if (silent < _timeout)
        continue;

      Fire(silent);
      return;
    }
  }

  private void Fire(TimeSpan silent)
  {
    _fired = true;
    var lastLine = _sink.LastLine;
    _sink.Write(
      $"[WATCHDOG] No log output for {silent.TotalSeconds:F0} s during startup. Last line: {lastLine}"
    );

    var path = Path.Combine(_sink.FolderPath, $"aethervk-hang-{Process.GetCurrentProcess().Id}.dmp");
    var written = false;
    try
    {
      written = _writeDump(path);
    }
    catch (Exception)
    {
      // a failed dump must not take the process down; the log lines above still record the stall
    }

    _sink.Write(written ? $"[WATCHDOG] Dump written: {path}" : "[WATCHDOG] No dump written");
    _sink.Flush(TimeSpan.FromSeconds(2));
  }
}

/// <summary>Windows minidump of the current process, through dbghelp.</summary>
internal static class MiniDump
{
  // MiniDumpWithThreadInfo: every thread's stack and context, no heap
  private const uint WithThreadInfo = 0x00001000;

  [DllImport("dbghelp.dll", SetLastError = true)]
  private static extern bool MiniDumpWriteDump(
    IntPtr process,
    uint processId,
    Microsoft.Win32.SafeHandles.SafeFileHandle file,
    uint dumpType,
    IntPtr exceptionParam,
    IntPtr userStreamParam,
    IntPtr callbackParam
  );

  public static bool TryWrite(string path)
  {
    if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
      return false;

    using (var file = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None))
    using (var process = Process.GetCurrentProcess())
    {
      return MiniDumpWriteDump(
        process.Handle,
        (uint)process.Id,
        file.SafeFileHandle,
        WithThreadInfo,
        IntPtr.Zero,
        IntPtr.Zero,
        IntPtr.Zero
      );
    }
  }
}
