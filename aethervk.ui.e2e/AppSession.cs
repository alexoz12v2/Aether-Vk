using System.Diagnostics;
using System.Net.Sockets;
using OpenQA.Selenium;
using OpenQA.Selenium.Appium;
using OpenQA.Selenium.Appium.Windows;

namespace AetherVk.UI.E2E;

/// <summary>
/// One running AetherVk process driven through WinAppDriver: <see cref="App"/> is the session on
/// the main window, <see cref="Desktop"/> the root session that sees the native file dialogs, the
/// popups (combo box drop-downs) and the desktop billboard windows.
/// </summary>
public sealed class AppSession : IDisposable
{
  public const string DriverUrl = "http://127.0.0.1:4723";

  private static readonly string DefaultDriverPath =
    @"C:\Program Files (x86)\Windows Application Driver\WinAppDriver.exe";

  private static Process? _driverProcess;

  public Process Process { get; }
  public WindowsDriver<WindowsElement> App { get; }
  public WindowsDriver<WindowsElement> Desktop { get; }
  public IntPtr MainWindow { get; }

  private AppSession(Process process, IntPtr mainWindow, WindowsDriver<WindowsElement> app,
    WindowsDriver<WindowsElement> desktop)
  {
    Process = process;
    MainWindow = mainWindow;
    App = app;
    Desktop = desktop;
  }

  /// <summary>Why the suite cannot run here (null when it can).</summary>
  public static string? UnavailableReason()
  {
    if (!OperatingSystem.IsWindows())
      return "WinAppDriver needs Windows";
    if (!File.Exists(AppPath()))
      return $"AetherVk.exe not found at {AppPath()} (build aethervk.ui-app in Release or set AETHERVK_E2E_APP)";
    if (!IsDriverListening() && !File.Exists(DriverPath()))
      return $"WinAppDriver not running and not found at {DriverPath()} (set AETHERVK_E2E_WINAPPDRIVER)";
    return null;
  }

  public static string AppPath()
  {
    var env = Environment.GetEnvironmentVariable("AETHERVK_E2E_APP");
    if (!string.IsNullOrWhiteSpace(env))
      return Path.GetFullPath(env);
    var repo = RepoRoot();
    foreach (var config in new[] { "Release", "Debug" })
    {
      var candidate = Path.Combine(repo, "aethervk.ui-app", "bin", config, "net10.0", "AetherVk.exe");
      if (File.Exists(candidate))
        return candidate;
    }
    return Path.Combine(repo, "aethervk.ui-app", "bin", "Release", "net10.0", "AetherVk.exe");
  }

  public static string RepoRoot()
  {
    var dir = new DirectoryInfo(AppContext.BaseDirectory);
    while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "AetherVk.sln")))
      dir = dir.Parent;
    return dir?.FullName ?? throw new InvalidOperationException("AetherVk.sln not found above the test output");
  }

  private static string DriverPath() =>
    Environment.GetEnvironmentVariable("AETHERVK_E2E_WINAPPDRIVER") is { Length: > 0 } p ? p : DefaultDriverPath;

  private static bool IsDriverListening()
  {
    try
    {
      using var client = new TcpClient();
      return client.ConnectAsync("127.0.0.1", 4723).Wait(TimeSpan.FromMilliseconds(500));
    }
    catch (Exception)
    {
      return false;
    }
  }

  private static void EnsureDriver()
  {
    if (IsDriverListening())
      return;
    // it exits on "Press ENTER" as soon as stdin reads EOF (as under dotnet test): hold a pipe open
    _driverProcess ??= Process.Start(new ProcessStartInfo(DriverPath())
    {
      UseShellExecute = false,
      CreateNoWindow = true,
      RedirectStandardInput = true,
    });
    var deadline = DateTime.UtcNow.AddSeconds(15);
    while (!IsDriverListening())
    {
      if (DateTime.UtcNow > deadline)
        throw new InvalidOperationException("WinAppDriver did not start listening on 127.0.0.1:4723");
      Thread.Sleep(250);
    }
  }

  /// <summary>
  /// Starts the app and waits past the splash for the main window. With <paramref name="logPath"/>
  /// its stdout / stderr (the C# console and the core's log callback) are written there.
  /// </summary>
  public static AppSession Launch(string? logPath = null, TimeSpan? startupTimeout = null)
  {
    EnsureDriver();
    var exe = AppPath();
    var process = new Process
    {
      StartInfo = new ProcessStartInfo(exe)
      {
        WorkingDirectory = Path.GetDirectoryName(exe)!,
        UseShellExecute = false,
        RedirectStandardOutput = logPath is not null,
        RedirectStandardError = logPath is not null,
      },
    };
    if (logPath is not null)
    {
      Directory.CreateDirectory(Path.GetDirectoryName(logPath)!);
      var log = TextWriter.Synchronized(new StreamWriter(logPath, append: false) { AutoFlush = true });
      process.OutputDataReceived += (_, e) => { if (e.Data is not null) log.WriteLine(e.Data); };
      process.ErrorDataReceived += (_, e) => { if (e.Data is not null) log.WriteLine("[stderr] " + e.Data); };
      process.Exited += (_, _) => log.Dispose();
      process.EnableRaisingEvents = true;
    }
    if (!process.Start())
      throw new InvalidOperationException($"could not start {exe}");
    if (logPath is not null)
    {
      process.BeginOutputReadLine();
      process.BeginErrorReadLine();
    }

    try
    {
      var main = WaitForMainWindow(process, startupTimeout ?? TimeSpan.FromSeconds(90));

      var appOptions = new AppiumOptions();
      appOptions.AddAdditionalCapability("appTopLevelWindow", main.ToString("x"));
      var app = new WindowsDriver<WindowsElement>(new Uri(DriverUrl), appOptions, TimeSpan.FromMinutes(2));

      var rootOptions = new AppiumOptions();
      rootOptions.AddAdditionalCapability("app", "Root");
      var desktop = new WindowsDriver<WindowsElement>(new Uri(DriverUrl), rootOptions, TimeSpan.FromMinutes(2));

      app.Manage().Window.Maximize();
      return new AppSession(process, main, app, desktop);
    }
    catch
    {
      if (!process.HasExited)
        process.Kill(entireProcessTree: true);
      throw;
    }
  }

  private static IntPtr WaitForMainWindow(Process process, TimeSpan timeout)
  {
    var deadline = DateTime.UtcNow + timeout;
    while (DateTime.UtcNow < deadline)
    {
      if (process.HasExited)
        throw new InvalidOperationException($"AetherVk exited during startup (code {process.ExitCode})");
      // the splash is "AetherVk Starting...", the main window "AetherVk" / "AetherVk — <save>"
      var main = Win32.TopLevelWindows(process.Id)
        .FirstOrDefault(w => w.Title.StartsWith("AetherVk", StringComparison.Ordinal)
                             && !w.Title.Contains("Starting", StringComparison.Ordinal));
      if (main is not null)
      {
        // let the layout, the viewport and the first frame settle before automation attaches
        Thread.Sleep(3000);
        return main.Handle;
      }
      Thread.Sleep(500);
    }
    throw new TimeoutException("the AetherVk main window did not appear");
  }

  public string WindowTitle => App.Title;

  public void Dispose()
  {
    try { App.Quit(); } catch (WebDriverException) { }
    try { Desktop.Quit(); } catch (WebDriverException) { }
    if (!Process.HasExited)
    {
      Process.CloseMainWindow();
      if (!Process.WaitForExit(10_000))
        Process.Kill(entireProcessTree: true);
    }
    Process.Dispose();
  }
}
