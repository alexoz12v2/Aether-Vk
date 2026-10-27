using System;
using System.Diagnostics;
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
      logSink.Flush(TimeSpan.FromSeconds(2));
    };
    TaskScheduler.UnobservedTaskException += (_, e) =>
    {
      logSink.Write($"[ERROR] Unobserved task exception: {e.Exception}");
    };

    // 1. Setup Microsoft Hosting
    var host = Host.CreateDefaultBuilder(args)
      .ConfigureServices(
        (context, services) =>
        {
          services.AddSingleton(logSink);
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
      // TODO: test on windows
      .With(
        new Win32PlatformOptions
        {
          RenderingMode =
          [
            Win32RenderingMode.Vulkan, // 1. Try vulkan first
            Win32RenderingMode.AngleEgl, // 2. then go to default, angle translates to Direct3D 11
            Win32RenderingMode.Software, // 3. CPU
          ],
        }
      )
      .LogToTrace()
      .UseReactiveUI();
}
