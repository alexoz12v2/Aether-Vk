using System;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using AetherVk.Utils;
using Avalonia;
using Avalonia.ReactiveUI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace AetherVk;

class Program
{
  // Initialization code. Don't use any Avalonia, third-party APIs or any
  // SynchronizationContext-reliant code before AppMain is called: things aren't initialized
  // yet and stuff might break.
  [STAThread]
  public static void Main(string[] args)
  {
    // Initialize X11 threading, necessary for multithreaded Xlib usage (we don't know whether
    // Avalonia does it, and even so, we don't want to rely on it)
    // Since we are on Avalonia 11 (and in the latest version at the time of writing, which is
    // 12.1), We don't have Wayland support with a IPlatformHandle, so we can safely assume for now
    // that we are on X11
    //
    // AvaloniaX11Platform actually does this, but we prefer doing it regardless
    // Furthermore, X11 docs says that you should call this before creating any other threads
    if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
    {
      try
      {
        _ = Logic.Services.PInvokeX11.XInitThreads();
      }
      catch (DllNotFoundException) { }
    }

    // Numbers read and print with a dot on every OS locale (an Italian Windows otherwise shows and
    // parses "0,5"); dates and the rest of the locale are kept.
    var culture = (CultureInfo)CultureInfo.CurrentCulture.Clone();
    culture.NumberFormat = CultureInfo.InvariantCulture.NumberFormat;
    CultureInfo.CurrentCulture = culture;
    CultureInfo.DefaultThreadCurrentCulture = culture;

    bool skipNative = false;
#if DEBUG
    skipNative = Array.IndexOf(args, "--skip-native") >= 0;
#endif

    // Log file first, so the session header and any crash below are recorded even in release
    var logSink = LogFileSink.CreateDefault();
    WriteSessionHeader(logSink, args);
    AppDomain.CurrentDomain.UnhandledException += (_, e) =>
    {
      logSink.Write($"[FATAL] Unhandled exception (terminating: {e.IsTerminating}): {e.ExceptionObject}");
      CrashReport.Write(logSink, "Unhandled exception", $"terminating: {e.IsTerminating}\n{e.ExceptionObject}");
    };
    TaskScheduler.UnobservedTaskException += (_, e) =>
    {
      logSink.Write($"[ERROR] Unobserved task exception: {e.Exception}");
    };

    // A startup that goes silent (a driver stalling in shader compilation, say) gets a dump after
    // this many seconds. AETHERVK_WATCHDOG_SECONDS=0 turns it off.
    var watchdogSeconds = 45;
    if (int.TryParse(Environment.GetEnvironmentVariable("AETHERVK_WATCHDOG_SECONDS"), out var configured))
      watchdogSeconds = configured;
    var watchdog = new StartupWatchdog(logSink, TimeSpan.FromSeconds(watchdogSeconds));
    if (watchdogSeconds > 0)
      watchdog.Start();

    // 1. Setup Microsoft Hosting
    var host = Host.CreateDefaultBuilder(args)
      .ConfigureServices(
        (context, services) =>
        {
          services.AddSingleton(logSink);
          services.AddSingleton(watchdog);
          services.AddCommonServices(skipNative);
          services.AddViewModels();
        }
      )
      .Build();

    // 2. Pass the host to the App so that it can solve services
    App.Host = host;

    // 3. start the host
    host.Start();

    // 4. Run Avalonia
    try
    {
      BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
    }
    finally
    {
      // 5. Ensure the host stops gracefully when Avalonia exits
      host.StopAsync().Wait();
      host.Dispose();
    }
  }

  private static void WriteSessionHeader(LogFileSink logSink, string[] args)
  {
    var version =
      typeof(Program).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
      ?? typeof(Program).Assembly.GetName().Version?.ToString()
      ?? "unknown";
#if DEBUG
    var build = "Debug";
#else
    var build = "Release";
#endif
    logSink.Write(
      $"[INFO] Aether-Vk {version} ({build}) started {DateTime.Now:O}, pid {Process.GetCurrentProcess().Id}"
    );
    logSink.Write(
      $"[INFO] OS: {RuntimeInformation.OSDescription} {RuntimeInformation.ProcessArchitecture}, {RuntimeInformation.FrameworkDescription}"
    );
    logSink.Write($"[INFO] Args: {string.Join(" ", args)}");
    logSink.Write($"[INFO] Log file: {logSink.FilePath}");
  }

  // Avalonia configuration, don't remove; also used by visual designer.
  public static AppBuilder BuildAvaloniaApp() =>
    AppBuilder
      .Configure<App>()
      .UsePlatformDetect()
      .WithInterFont() // use Inter as default font everywhere
      .With(
        new X11PlatformOptions
        {
          RenderingMode =
          [
            // bad bugwith nvidia driver: deadlock on thread 31 between glXMakeContextCurrent and any
            // "heavy" vulkan function (end cmd buffer, present, ..). Egl and Vulkan do not support
            // consistently window transparency (tested on wayland gnome-shell with xwayland), so
            // software is the only way
            // X11RenderingMode.Glx,
            X11RenderingMode.Software,
          ],
        }
      )
      .With(
        new Win32PlatformOptions
        {
          RenderingMode =
          [
            // black screen / crash on hybrid-GPU laptops (RTX dGPU + iGPU) with Vulkan: a second
            // Vulkan device next to the core's, possibly on the other adapter. The UI is light
            // enough for the CPU, and the 3D viewport is drawn by the core, so software only
            // Win32RenderingMode.Vulkan,
            // Win32RenderingMode.AngleEgl,
            Win32RenderingMode.Software,
          ],
        }
      )
      .LogToTrace()
      .UseReactiveUI();
}
