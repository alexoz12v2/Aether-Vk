using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Threading.Tasks;

namespace AetherVk.Logic.Services;

public sealed class LocalStorageService : ILocalStorageService
{
  public string PersistentDirectory { get; }
  public string SessionDirectory { get; }
  public string SessionId { get; }

  /// <summary>Name of the file in each session directory naming its owner process.</summary>
  public const string PidFileName = "pid";

  public LocalStorageService()
    : this(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aethervk")) { }

  /// <summary>Storage rooted at <paramref name="persistentDirectory"/> (tests).</summary>
  public LocalStorageService(string persistentDirectory)
  {
    PersistentDirectory = persistentDirectory;

    SessionId = Guid.NewGuid().ToString("N");
    SessionDirectory = Path.Combine(PersistentDirectory, SessionId);

    EnsureDirectories();
    WritePidFile();
    CleanupOldSessions();
  }

  /// <summary>
  /// <c>&lt;pid&gt; &lt;process start, UTC ticks&gt;</c>: a later run deletes this session directory
  /// once that process is gone, whatever ended it (a crash or <c>kill -9</c> never reaches
  /// <see cref="Dispose"/>). The start time guards against a recycled pid.
  /// </summary>
  private void WritePidFile()
  {
    try
    {
      using var me = Process.GetCurrentProcess();
      File.WriteAllText(
        Path.Combine(SessionDirectory, PidFileName),
        $"{me.Id} {me.StartTime.ToUniversalTime().Ticks}");
    }
    catch (Exception e) when (e is IOException or UnauthorizedAccessException or InvalidOperationException)
    {
    }
  }

  /// <summary>Start time (UTC) of process <paramref name="pid"/>, null when there is none.</summary>
  public static DateTime? ProcessStartTimeUtc(int pid)
  {
    try
    {
      using var p = Process.GetProcessById(pid);
      return p.HasExited ? null : p.StartTime.ToUniversalTime();
    }
    catch (Exception e) when (e is ArgumentException or InvalidOperationException or System.ComponentModel.Win32Exception)
    {
      return null;
    }
  }

  /// <summary>
  /// Whether session directory <paramref name="sessionDir"/> belongs to no running instance: its
  /// pid file names a process that is gone (or another process now holding that pid). Legacy
  /// directories without a pid file count as abandoned after a day.
  /// </summary>
  public static bool IsAbandoned(string sessionDir, DateTime nowUtc, Func<int, DateTime?> processStartUtc)
  {
    var pidFile = Path.Combine(sessionDir, PidFileName);
    if (!File.Exists(pidFile))
      return Directory.GetCreationTimeUtc(sessionDir) < nowUtc.AddDays(-1);
    try
    {
      var parts = File.ReadAllText(pidFile).Trim().Split(' ');
      int pid = int.Parse(parts[0], CultureInfo.InvariantCulture);
      long ticks = long.Parse(parts[1], CultureInfo.InvariantCulture);
      var start = processStartUtc(pid);
      return start is null || Math.Abs((start.Value - new DateTime(ticks, DateTimeKind.Utc)).TotalSeconds) > 1;
    }
    catch (Exception e) when (e is FormatException or OverflowException or IndexOutOfRangeException or IOException)
    {
      return true;
    }
  }

  private void EnsureDirectories()
  {
    if (!Directory.Exists(PersistentDirectory))
    {
      Directory.CreateDirectory(PersistentDirectory);
    }
    if (!Directory.Exists(SessionDirectory))
    {
      Directory.CreateDirectory(SessionDirectory);
    }
  }

  private void CleanupOldSessions()
  {
    try
    {
      if (!Directory.Exists(PersistentDirectory))
        return;

      var dirs = Directory.GetDirectories(PersistentDirectory);

      foreach (var dir in dirs)
      {
        var dirInfo = new DirectoryInfo(dir);
        // Skip the current session directory
        if (dirInfo.Name == SessionId)
          continue;

        // Check if it matches GUID format (length 32 for "N" format)
        if (dirInfo.Name.Length == 32 && Guid.TryParseExact(dirInfo.Name, "N", out _))
        {
          if (IsAbandoned(dir, DateTime.UtcNow, ProcessStartTimeUtc))
          {
            try
            {
              dirInfo.Delete(true);
            }
            catch
            {
              // Ignore if we can't delete right now (e.g. file in use)
            }
          }
        }
      }
    }
    catch
    {
      // Logging can be added here if needed, but startup cleanup should not crash the app
    }
  }

  public string GetPersistentPath(string relativePath)
  {
    var path = Path.Combine(PersistentDirectory, relativePath);
    EnsureParentDirectoryExists(path);
    return path;
  }

  public string GetSessionPath(string relativePath)
  {
    var path = Path.Combine(SessionDirectory, relativePath);
    EnsureParentDirectoryExists(path);
    return path;
  }

  /// <summary>
  /// Where downloads (SPK kernels) go: <c>&lt;session&gt;/spk/</c>, deleted with the session. A save
  /// copies the ones it needs into its own <c>spk/</c> folder (docs/savefile.md).
  /// </summary>
  public string GetDownloadsPath(string fileName) => GetSessionPath(Path.Combine("spk", fileName));

  private void EnsureParentDirectoryExists(string path)

  {
    var dir = Path.GetDirectoryName(path);
    if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
    {
      Directory.CreateDirectory(dir);
    }
  }

  public async Task SavePersistentAsync(string relativePath, byte[] data)
  {
    var path = GetPersistentPath(relativePath);
    using var fs = new FileStream(
      path,
      FileMode.Create,
      FileAccess.Write,
      FileShare.None,
      4096,
      true
    );
    await fs.WriteAsync(data, 0, data.Length);
  }

  public async Task SavePersistentAsync(string relativePath, Stream data)
  {
    var path = GetPersistentPath(relativePath);
    using var fs = new FileStream(
      path,
      FileMode.Create,
      FileAccess.Write,
      FileShare.None,
      4096,
      true
    );
    data.Position = 0;
    await data.CopyToAsync(fs);
  }

  public async Task SaveSessionAsync(string relativePath, byte[] data)
  {
    var path = GetSessionPath(relativePath);
    using var fs = new FileStream(
      path,
      FileMode.Create,
      FileAccess.Write,
      FileShare.None,
      4096,
      true
    );
    await fs.WriteAsync(data, 0, data.Length);
  }

  public async Task SaveSessionAsync(string relativePath, Stream data)
  {
    var path = GetSessionPath(relativePath);
    using var fs = new FileStream(
      path,
      FileMode.Create,
      FileAccess.Write,
      FileShare.None,
      4096,
      true
    );
    data.Position = 0;
    await data.CopyToAsync(fs);
  }

  public void Dispose()
  {
    // Clean up downloads directory for this session
    try
    {
      var homeDir = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
      var downloadsDir = Path.Combine(homeDir, "Downloads", $"AetherVk_{SessionId}");
      if (Directory.Exists(downloadsDir))
      {
        Directory.Delete(downloadsDir, true);
      }
    }
    catch
    {
      // Ignore cleanup failures during shutdown
    }

    // Clean up current session directory
    try
    {
      if (Directory.Exists(SessionDirectory))
      {
        Directory.Delete(SessionDirectory, true);
      }
    }
    catch
    {
      // Ignore cleanup failures during shutdown
    }
  }
}
