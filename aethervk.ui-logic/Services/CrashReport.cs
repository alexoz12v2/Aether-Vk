using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace AetherVk.Logic.Services;

/// <summary>
/// A crash report next to the run logs (<c>aethervk-crash-&lt;stamp&gt;-&lt;pid&gt;.txt</c>): what
/// crashed, the version and OS, and the last <see cref="LogFileSink.RecentLineCount"/> log lines.
/// Written synchronously, on the crashing thread, because the process dies right after: an
/// unhandled .NET exception, or a native engine panic (the core aborts after its panic hook sends
/// a <see cref="NativePanicPrefix"/> line through the logger callback). The reviewer report of
/// 2026-10-29 ("it crashes after a few minutes") came with nothing to read; the next one carries
/// this file.
/// </summary>
public static class CrashReport
{
  /// <summary>The core's panic line prefix (<c>PANIC_LOG_PREFIX</c> in the cdylib).</summary>
  public const string NativePanicPrefix = "[PANIC]";

  private const string Prefix = "aethervk-crash-";

  /// <summary>Newest crash reports kept in the folder, this one included.</summary>
  public const int KeptReports = 10;

  /// <summary>
  /// Writes the report and logs its path, then flushes the run log. Never throws: returns the
  /// path, or null when nothing could be written.
  /// </summary>
  public static string? Write(LogFileSink sink, string what, string details, DateTime? now = null)
  {
    try
    {
      var stamp = now ?? DateTime.Now;
      var dir = sink.FolderPath;
      Directory.CreateDirectory(dir);
      RemoveOld(dir);
      int pid = Process.GetCurrentProcess().Id;
      var path = Path.Combine(dir, $"{Prefix}{stamp:yyyyMMdd-HHmmss}-{pid}.txt");
      var text = new StringBuilder()
        .AppendLine($"Aether-Vk crash report: {what}")
        .AppendLine($"Time: {stamp:O}, pid {pid}")
        .AppendLine($"Version: {Version()}")
        .AppendLine($"OS: {RuntimeInformation.OSDescription} {RuntimeInformation.ProcessArchitecture}, {RuntimeInformation.FrameworkDescription}")
        .AppendLine($"Run log: {sink.FilePath}")
        .AppendLine()
        .AppendLine("== Details")
        .AppendLine(details)
        .AppendLine()
        .AppendLine($"== Last {LogFileSink.RecentLineCount} log lines");
      foreach (var line in sink.RecentLines())
        text.AppendLine(line);
      File.WriteAllText(path, text.ToString(), new UTF8Encoding(false));
      sink.Write($"[FATAL] {what}: crash report written to {path}");
      sink.Flush(TimeSpan.FromSeconds(2));
      return path;
    }
    catch (Exception)
    {
      // a crash report must never be the second crash
      try
      {
        sink.Flush(TimeSpan.FromSeconds(2));
      }
      catch (Exception) { }
      return null;
    }
  }

  /// <summary>Set to "1" by the end-to-end suite on the app it launches.</summary>
  public const string E2EEnvVar = "AETHERVK_E2E";

  /// <summary>"1" (with <see cref="E2EEnvVar"/>) crashes the app on purpose once its main window is up.</summary>
  public const string TestCrashEnvVar = "AETHERVK_TEST_CRASH";

  /// <summary>The message of the exception the test crash throws (in the report's details).</summary>
  public const string TestCrashMessage = "test crash requested through " + TestCrashEnvVar;

  /// <summary>
  /// A test crash is asked for: <see cref="TestCrashEnvVar"/> and <see cref="E2EEnvVar"/> are
  /// both "1". Either alone does nothing, so a stray variable on a user's machine never crashes it.
  /// </summary>
  public static bool TestCrashRequested(Func<string, string?> getEnvironmentVariable) =>
    getEnvironmentVariable(E2EEnvVar) == "1" && getEnvironmentVariable(TestCrashEnvVar) == "1";

  /// <summary>
  /// When <see cref="TestCrashRequested"/>, throws an unhandled exception on a background thread
  /// after <paramref name="delay"/>: the path a real crash takes (the unhandled-exception handler
  /// writes the report, then the process dies). Returns whether it was armed.
  /// </summary>
  public static bool StartTestCrashIfRequested(Func<string, string?> getEnvironmentVariable, TimeSpan delay)
  {
    if (!TestCrashRequested(getEnvironmentVariable))
      return false;
    var thread = new System.Threading.Thread(() =>
    {
      System.Threading.Thread.Sleep(delay);
      throw new InvalidOperationException(TestCrashMessage);
    })
    { IsBackground = true, Name = "TestCrash" };
    thread.Start();
    return true;
  }

  private static string Version()
  {
    var asm = System.Reflection.Assembly.GetEntryAssembly() ?? typeof(CrashReport).Assembly;
    return asm.GetCustomAttributes(typeof(System.Reflection.AssemblyInformationalVersionAttribute), false)
             .OfType<System.Reflection.AssemblyInformationalVersionAttribute>().FirstOrDefault()?.InformationalVersion
           ?? asm.GetName().Version?.ToString() ?? "unknown";
  }

  private static void RemoveOld(string dir)
  {
    // names start with the timestamp: the newest sort last
    foreach (var old in Directory.GetFiles(dir, Prefix + "*.txt")
               .OrderByDescending(f => f, StringComparer.Ordinal).Skip(KeptReports - 1))
    {
      try
      {
        File.Delete(old);
      }
      catch (Exception) { }
    }
  }
}
