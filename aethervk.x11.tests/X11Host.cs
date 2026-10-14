using System.Diagnostics;
using System.Globalization;
using System.Text.RegularExpressions;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Threading;

namespace AetherVk.X11Tests;

/// <summary>Minimal Avalonia application for the X11 tests (no DI host, no main view).</summary>
public sealed class X11TestApp : Application
{
  public static readonly ManualResetEventSlim Ready = new();

  public override void OnFrameworkInitializationCompleted()
  {
    base.OnFrameworkInitializationCompleted();
    Ready.Set();
  }
}

/// <summary>
/// The real Avalonia X11 platform on its own UI thread (started once per test process), plus the
/// X tools the tests drive and inspect it with (<c>xdotool</c>, <c>xwininfo</c>, <c>xprop</c>,
/// <c>import</c>).
/// </summary>
public static class X11Host
{
  private static readonly object Gate = new();
  private static bool _started;

  /// <summary>Why the tests cannot run here, or null when they can.</summary>
  public static string? SkipReason
  {
    get
    {
      if (Environment.GetEnvironmentVariable("AETHERVK_X11_E2E") != "1")
        return "X11 end-to-end tests run through scripts/x11_e2e.sh (AETHERVK_X11_E2E=1)";
      if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DISPLAY")))
        return "no DISPLAY";
      foreach (var tool in new[] { "xdotool", "xwininfo", "xprop" })
        if (!File.Exists($"/usr/bin/{tool}"))
          return $"{tool} not installed";
      return null;
    }
  }

  public static void EnsureStarted()
  {
    lock (Gate)
    {
      if (_started)
        return;
      _started = true;
      var t = new Thread(() =>
      {
        AppBuilder
          .Configure<X11TestApp>()
          .UsePlatformDetect()
          .With(new X11PlatformOptions { RenderingMode = [X11RenderingMode.Software] })
          .StartWithClassicDesktopLifetime([], ShutdownMode.OnExplicitShutdown);
      })
      {
        IsBackground = true,
        Name = "Avalonia X11 UI",
      };
      t.Start();
      if (!X11TestApp.Ready.Wait(TimeSpan.FromSeconds(30)))
        throw new TimeoutException("Avalonia X11 platform did not start");
    }
  }

  public static T Ui<T>(Func<T> f) => Dispatcher.UIThread.InvokeAsync(f).GetAwaiter().GetResult();

  public static void Ui(Action f) => Dispatcher.UIThread.InvokeAsync(f).GetAwaiter().GetResult();

  /// <summary>Lets the UI thread and the X server catch up.</summary>
  public static void Settle(int ms = 150)
  {
    Thread.Sleep(ms);
    Ui(() => Dispatcher.UIThread.RunJobs());
  }

  /// <summary>Polls <paramref name="cond"/> (evaluated on the test thread) for up to 5 s.</summary>
  public static bool WaitFor(Func<bool> cond, int timeoutMs = 5000)
  {
    var sw = Stopwatch.StartNew();
    while (sw.ElapsedMilliseconds < timeoutMs)
    {
      if (cond())
        return true;
      Thread.Sleep(50);
    }
    return cond();
  }

  public static string Run(string exe, string args)
  {
    var psi = new ProcessStartInfo(exe, args)
    {
      RedirectStandardOutput = true,
      RedirectStandardError = true,
      UseShellExecute = false,
    };
    using var p = Process.Start(psi)!;
    string output = p.StandardOutput.ReadToEnd();
    p.WaitForExit(10_000);
    return output;
  }

  public static string Hex(nint xid) => "0x" + ((long)xid).ToString("x", CultureInfo.InvariantCulture);

  /// <summary><c>xwininfo -all</c> fields of a window.</summary>
  public sealed record WinInfo(int X, int Y, int Width, int Height, bool Viewable, bool OverrideRedirect);

  public static WinInfo XWinInfo(nint xid)
  {
    string s = Run("xwininfo", $"-all -id {Hex(xid)}");
    int Field(string name) =>
      int.Parse(Regex.Match(s, name + @":\s+(-?\d+)").Groups[1].Value, CultureInfo.InvariantCulture);
    return new WinInfo(
      Field("Absolute upper-left X"),
      Field("Absolute upper-left Y"),
      Field("Width"),
      Field("Height"),
      s.Contains("Map State: IsViewable"),
      Regex.IsMatch(s, @"Override Redirect State:\s+yes"));
  }

  public static string XProp(nint xid, string prop) => Run("xprop", $"-id {Hex(xid)} {prop}");

  public static void XDo(string args) => Run("xdotool", args);

  /// <summary>The composited colour of a desktop pixel (needs a compositing manager for alpha).</summary>
  public static (int R, int G, int B)? RootPixel(int x, int y)
  {
    if (!File.Exists("/usr/bin/import"))
      return null;
    string s = Run("import", $"-window root -crop 1x1+{x}+{y} txt:-");
    var m = Regex.Match(s, @"\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)");
    if (!m.Success)
      return null;
    double[] v = [.. Enumerable.Range(1, 3).Select(g => double.Parse(m.Groups[g].Value, CultureInfo.InvariantCulture))];
    // ImageMagick Q16 prints 16-bit channels
    double k = v.Max() > 255 ? 257.0 : 1.0;
    return ((int)Math.Round(v[0] / k), (int)Math.Round(v[1] / k), (int)Math.Round(v[2] / k));
  }

  /// <summary>A compositing manager owns the screen's <c>_NET_WM_CM_S0</c> selection.</summary>
  public static bool CompositorRunning() =>
    Run("pgrep", "-x xcompmgr").Trim().Length > 0 || Run("pgrep", "-x picom").Trim().Length > 0;
}
