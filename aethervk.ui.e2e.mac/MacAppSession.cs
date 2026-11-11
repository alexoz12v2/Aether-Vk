using System.Collections.Concurrent;
using System.Diagnostics;
using System.Net.Sockets;
using OpenQA.Selenium;
using OpenQA.Selenium.Appium;
using OpenQA.Selenium.Appium.Mac;

namespace AetherVk.UI.E2E.Mac;

/// <summary>
/// One running AetherVk.app driven through Appium's Mac2 driver (XCTest / WebDriverAgentMac).
/// <para>
/// Default ("process") mode: the bundle's binary is started with <see cref="System.Diagnostics.Process"/>
/// (stdout / stderr captured, <c>AETHERVK_E2E=1</c>, <c>AETHERVK_UI_CULTURE</c>), then a Mac2
/// session attaches to the running app by <c>appium:bundleId</c> with <c>appium:noReset</c> and
/// <c>appium:skipAppKill</c>, so Mac2 neither relaunches nor kills it.
/// </para>
/// <para>
/// Fallback (<c>AETHERVK_E2E_MAC_LAUNCH=driver</c>), for when the attach relaunches the app: Mac2
/// starts it with <c>appium:arguments</c> / <c>appium:environment</c>, the process is found by
/// name, and its log is the app's own <c>LogFileSink</c> file
/// (<c>~/Library/Logs/Aether-Vk/aethervk-&lt;stamp&gt;-&lt;pid&gt;.log</c>) tailed into <see cref="Lines"/>.
/// </para>
/// </summary>
public sealed class MacAppSession : IDisposable
{
  public const string DriverUrl = "http://127.0.0.1:4723";

  /// <summary>The SideBySide bundle id of <c>scripts/package_macos.sh osx-arm64 Release SideBySide</c>.</summary>
  public const string DefaultBundleId = "com.example.AetherVk.sxs.1-0-1.release";

  /// <summary>The bundle that script writes, relative to the repo.</summary>
  public const string DefaultBundle = "bin/publish/AetherVk-SxS-1.0.1-Release.app";

  public const string ExecutableName = "AetherVk";

  private const int MaxLines = 20_000;

  private static Process? _appiumProcess;

  public MacDriver Driver { get; }

  /// <summary>Our child in process mode; the process found by name in driver mode (no exit code then).</summary>
  public Process Process { get; }

  public bool LaunchedByDriver { get; }

  public ConcurrentQueue<(DateTime At, string Line)> Lines { get; }

  private readonly LogTail? _tail;

  private MacAppSession(MacDriver driver, Process process, bool byDriver, ConcurrentQueue<(DateTime, string)> lines, LogTail? tail)
  {
    Driver = driver;
    Process = process;
    LaunchedByDriver = byDriver;
    Lines = lines;
    _tail = tail;
  }

  public bool IsAlive
  {
    get
    {
      try
      {
        return !Process.HasExited;
      }
      catch (InvalidOperationException)
      {
        return false;
      }
    }
  }

  public static string BundleId() =>
    Environment.GetEnvironmentVariable("AETHERVK_E2E_BUNDLE_ID") is { Length: > 0 } id ? id : DefaultBundleId;

  public static bool DriverLaunchMode() =>
    string.Equals(Environment.GetEnvironmentVariable("AETHERVK_E2E_MAC_LAUNCH"), "driver", StringComparison.OrdinalIgnoreCase);

  /// <summary>The .app (AETHERVK_E2E_APP may name the .app or its binary).</summary>
  public static string AppPath()
  {
    var env = Environment.GetEnvironmentVariable("AETHERVK_E2E_APP");
    if (!string.IsNullOrWhiteSpace(env))
      return Path.GetFullPath(env);
    return Path.Combine(Shared.E2ESteps.RepoRoot(), DefaultBundle);
  }

  public static string Executable()
  {
    var app = AppPath().TrimEnd('/');
    return app.EndsWith(".app", StringComparison.OrdinalIgnoreCase)
      ? Path.Combine(app, "Contents", "MacOS", ExecutableName)
      : app;
  }

  /// <summary>The LogFileSink folder on macOS.</summary>
  public static string LogDirectory() =>
    Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Library", "Logs", "Aether-Vk");

  /// <summary>Why the suite cannot run here (null when it can).</summary>
  public static string? UnavailableReason()
  {
    if (!OperatingSystem.IsMacOS())
      return "the Mac2 driver needs macOS";
    if (!File.Exists(Executable()))
      return $"{Executable()} not found (scripts/package_macos.sh osx-arm64 Release SideBySide, or set AETHERVK_E2E_APP)";
    if (!IsDriverListening() && AppiumCommand() is null)
      return "Appium is not listening on 127.0.0.1:4723 and `appium` is not on PATH (npm i -g appium; appium driver install mac2)";
    return null;
  }

  private static string? AppiumCommand()
  {
    if (Environment.GetEnvironmentVariable("AETHERVK_E2E_APPIUM") is { Length: > 0 } explicitPath)
      return File.Exists(explicitPath) ? explicitPath : null;
    var path = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
    // dotnet test may run without the shell's PATH: the usual npm global bins too
    var dirs = path.Split(':', StringSplitOptions.RemoveEmptyEntries)
      .Concat(["/opt/homebrew/bin", "/usr/local/bin"]);
    return dirs.Select(d => Path.Combine(d, "appium")).FirstOrDefault(File.Exists);
  }

  public static bool IsDriverListening()
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

  /// <summary>Whether the server on 127.0.0.1:4723 answers <c>GET /status</c> with 200 (two
  /// consecutive answers half a second apart, so a server shutting down is not taken).</summary>
  public static bool IsDriverHealthy()
  {
    using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(2) };
    for (int i = 0; i < 2; i++)
    {
      try
      {
        if (!http.GetAsync(DriverUrl + "/status").GetAwaiter().GetResult().IsSuccessStatusCode)
          return false;
      }
      catch (Exception)
      {
        return false;
      }
      if (i == 0)
        Thread.Sleep(500);
    }
    return true;
  }

  /// <summary>Starts <c>appium</c> on 127.0.0.1:4723 when nothing listens there (its log next to the artifacts).</summary>
  public static void EnsureAppium()
  {
    // the server a previous `dotnet test` started dies with that test host (its stdout pipe
    // closes) and may still hold the port for a few seconds: a session sent to it ends with
    // "The response ended prematurely". Use a listener only once it answers /status; wait for a
    // dying one to let go of the port, then start our own.
    var settle = DateTime.UtcNow.AddSeconds(30);
    while (IsDriverListening())
    {
      if (IsDriverHealthy())
        return;
      if (DateTime.UtcNow > settle)
        throw new InvalidOperationException("something listens on 127.0.0.1:4723 but does not answer GET /status");
      Thread.Sleep(500);
    }
    var cmd = AppiumCommand() ?? throw new InvalidOperationException("appium not found");
    var log = Path.Combine(Shared.E2ESteps.RepoRoot(), "TestResults", "e2e-mac", "appium.log");
    Directory.CreateDirectory(Path.GetDirectoryName(log)!);
    if (_appiumProcess is null || _appiumProcess.HasExited)
    {
      var psi = new ProcessStartInfo(cmd)
      {
        UseShellExecute = false,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        ArgumentList = { "--address", "127.0.0.1", "--port", "4723" },
      };
      // node next to appium (a Homebrew / nvm install keeps them together)
      psi.Environment["PATH"] = Path.GetDirectoryName(cmd) + ":" + (Environment.GetEnvironmentVariable("PATH") ?? "/usr/bin:/bin");
      var writer = TextWriter.Synchronized(new StreamWriter(log, append: true) { AutoFlush = true });
      _appiumProcess = Process.Start(psi) ?? throw new InvalidOperationException("appium did not start");
      _appiumProcess.OutputDataReceived += (_, e) => { if (e.Data is not null) writer.WriteLine(e.Data); };
      _appiumProcess.ErrorDataReceived += (_, e) => { if (e.Data is not null) writer.WriteLine("[stderr] " + e.Data); };
      _appiumProcess.BeginOutputReadLine();
      _appiumProcess.BeginErrorReadLine();
    }
    var deadline = DateTime.UtcNow.AddSeconds(45);
    while (!IsDriverListening() || !IsDriverHealthy())
    {
      if (DateTime.UtcNow > deadline)
        throw new InvalidOperationException($"appium did not start listening on 127.0.0.1:4723 (log: {log})");
      Thread.Sleep(500);
    }
  }

  private static Dictionary<string, string> AppEnvironment(string? culture, bool diagnosticPanels)
  {
    var env = new Dictionary<string, string> { ["AETHERVK_E2E"] = "1" };
    if (culture is not null)
      env["AETHERVK_UI_CULTURE"] = culture;
    if (diagnosticPanels)
      env["AETHERVK_DIAGNOSTIC_PANELS"] = "1";
    return env;
  }

  private static AppiumOptions Options(bool attach, Dictionary<string, string>? environment)
  {
    var o = new AppiumOptions { AutomationName = "Mac2" };
    o.PlatformName = "mac";
    o.AddAdditionalAppiumOption("bundleId", BundleId());
    if (attach)
    {
      // the app is already running: neither restart it nor kill it at the session's end
      o.AddAdditionalAppiumOption("noReset", true);
      o.AddAdditionalAppiumOption("skipAppKill", true);
    }
    else
    {
      o.AddAdditionalAppiumOption("arguments", Array.Empty<string>());
      o.AddAdditionalAppiumOption("environment", environment ?? new Dictionary<string, string>());
    }
    o.AddAdditionalAppiumOption("newCommandTimeout", 600);
    return o;
  }

  /// <summary>
  /// Starts the app and waits past the splash for the main window. With <paramref name="logPath"/>
  /// the app's lines are written there (process mode: stdout / stderr; driver mode: the log file tail).
  /// </summary>
  public static MacAppSession Launch(string? logPath = null, string? culture = null, bool diagnosticPanels = false,
    TimeSpan? startupTimeout = null)
  {
    EnsureAppium();
    var lines = new ConcurrentQueue<(DateTime, string)>();
    TextWriter? log = null;
    if (logPath is not null)
    {
      Directory.CreateDirectory(Path.GetDirectoryName(logPath)!);
      log = TextWriter.Synchronized(new StreamWriter(logPath, append: false) { AutoFlush = true });
    }
    void Keep(string line)
    {
      log?.WriteLine(line);
      lines.Enqueue((DateTime.UtcNow, line));
      while (lines.Count > MaxLines)
        lines.TryDequeue(out _);
    }

    var env = AppEnvironment(culture, diagnosticPanels);
    var timeout = startupTimeout ?? TimeSpan.FromSeconds(90);
    if (DriverLaunchMode())
    {
      var started = DateTime.Now;
      var driver = new MacDriver(new Uri(DriverUrl), Options(attach: false, env), TimeSpan.FromMinutes(3));
      try
      {
        var process = Shared.E2EWait.For(() => FindAppProcess(started), timeout, "the AetherVk process");
        var tail = new LogTail(LogDirectory(), process.Id, Keep);
        WaitForMainWindow(driver, process, timeout);
        return new MacAppSession(driver, process, true, lines, tail);
      }
      catch
      {
        try { driver.Quit(); } catch (WebDriverException) { }
        throw;
      }
    }

    var exe = Executable();
    var p = new Process
    {
      StartInfo = new ProcessStartInfo(exe)
      {
        WorkingDirectory = Path.GetDirectoryName(exe)!,
        UseShellExecute = false,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
      },
      EnableRaisingEvents = true,
    };
    foreach (var (k, v) in env)
      p.StartInfo.Environment[k] = v;
    p.OutputDataReceived += (_, e) => { if (e.Data is not null) Keep(e.Data); };
    p.ErrorDataReceived += (_, e) => { if (e.Data is not null) Keep("[stderr] " + e.Data); };
    p.Exited += (_, _) => log?.Dispose();
    if (!p.Start())
      throw new InvalidOperationException($"could not start {exe}");
    p.BeginOutputReadLine();
    p.BeginErrorReadLine();
    try
    {
      // let it register with the window server before XCTest looks it up by bundle id
      Thread.Sleep(2000);
      if (p.HasExited)
        throw new InvalidOperationException($"AetherVk exited during startup (code {p.ExitCode})");
      var driver = new MacDriver(new Uri(DriverUrl), Options(attach: true, null), TimeSpan.FromMinutes(3));
      try
      {
        WaitForMainWindow(driver, p, timeout);
      }
      catch
      {
        try { driver.Quit(); } catch (WebDriverException) { }
        throw;
      }
      if (p.HasExited)
        throw new InvalidOperationException("the Mac2 attach ended the launched app (it relaunched it?): run with AETHERVK_E2E_MAC_LAUNCH=driver");
      return new MacAppSession(driver, p, false, lines, null);
    }
    catch
    {
      if (!p.HasExited)
        p.Kill(entireProcessTree: true);
      throw;
    }
  }

  /// <summary>The newest AetherVk process started after <paramref name="after"/> (driver mode).</summary>
  private static Process? FindAppProcess(DateTime after)
  {
    foreach (var p in Process.GetProcessesByName(ExecutableName).OrderByDescending(SafeStart))
      if (SafeStart(p) >= after.AddSeconds(-2))
        return p;
    return null;

    static DateTime SafeStart(Process p)
    {
      try { return p.StartTime; }
      catch (Exception) { return DateTime.MinValue; }
    }
  }

  /// <summary>The main window: its title starts with "AetherVk" and is not the splash's "Starting".</summary>
  public static IWebElement? MainWindowOf(MacDriver driver) =>
    ((ISearchContext)driver).FindElements(By.XPath("//XCUIElementTypeWindow"))
      .FirstOrDefault(w =>
      {
        // "AetherVk — <document>"; not the splash ("Starting") nor the other top-level windows that
        // share the prefix, e.g. the FITS popup "AetherVk billboard" (listed before the main window)
        string title;
        try
        {
          title = w.GetAttribute("title") ?? string.Empty;
        }
        catch (WebDriverException)
        {
          return false; // gone meanwhile (a full-screen or tiling transition replaces the window element)
        }
        return (title == "AetherVk" || title.StartsWith("AetherVk —", StringComparison.Ordinal))
               && !title.Contains("Starting", StringComparison.Ordinal);
      });

  private static void WaitForMainWindow(MacDriver driver, Process process, TimeSpan timeout)
  {
    var deadline = DateTime.UtcNow + timeout;
    while (DateTime.UtcNow < deadline)
    {
      if (process.HasExited)
        throw new InvalidOperationException("AetherVk exited during startup");
      try
      {
        if (MainWindowOf(driver) is not null)
        {
          // the layout, the viewport and the first frame settle before the scenario starts
          Thread.Sleep(3000);
          return;
        }
      }
      catch (WebDriverException)
      {
      }
      Thread.Sleep(500);
    }
    throw new TimeoutException("the AetherVk main window did not appear");
  }

  public void Dispose()
  {
    try { Driver.Quit(); } catch (WebDriverException) { }
    if (IsAlive)
    {
      try
      {
        // SIGTERM first: the app's own shutdown (the Vulkan teardown runs on the main thread)
        Process.Start("/bin/kill", ["-TERM", Process.Id.ToString(System.Globalization.CultureInfo.InvariantCulture)])?.WaitForExit(5000);
      }
      catch (Exception)
      {
      }
      if (!Process.WaitForExit(10_000))
        Process.Kill(entireProcessTree: true);
    }
    _tail?.Dispose();
    Process.Dispose();
  }
}

/// <summary>Follows the app's LogFileSink file (driver mode: stdout is not ours) into a line callback.</summary>
internal sealed class LogTail : IDisposable
{
  private readonly CancellationTokenSource _stop = new();
  private readonly Thread _thread;

  public LogTail(string directory, int pid, Action<string> line)
  {
    _thread = new Thread(() => Run(directory, pid, line)) { IsBackground = true, Name = "LogTail" };
    _thread.Start();
  }

  private void Run(string directory, int pid, Action<string> line)
  {
    string? file = null;
    while (!_stop.IsCancellationRequested && file is null)
    {
      file = Directory.Exists(directory)
        ? Directory.GetFiles(directory, $"aethervk-*-{pid}.log").FirstOrDefault()
        : null;
      if (file is null)
        _stop.Token.WaitHandle.WaitOne(250);
    }
    if (file is null)
      return;
    using var stream = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
    using var reader = new StreamReader(stream);
    while (!_stop.IsCancellationRequested)
    {
      var l = reader.ReadLine();
      if (l is null)
        _stop.Token.WaitHandle.WaitOne(200);
      else
        line(l);
    }
  }

  public void Dispose()
  {
    _stop.Cancel();
    _thread.Join(2000);
  }
}
