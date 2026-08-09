using System;
using System.Runtime.InteropServices;
using AetherVk.Utils;
using Avalonia;
using Avalonia.ReactiveUI;
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

    // 1. Setup Microsoft Hosting
    var host = Host.CreateDefaultBuilder(args)
      .ConfigureServices(
        (context, services) =>
        {
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

  // Avalonia configuration, don't remove; also used by visual designer.
  public static AppBuilder BuildAvaloniaApp() =>
    AppBuilder.Configure<App>()
      .UsePlatformDetect()
      .WithInterFont() // use Inter as default font everywhere
      .LogToTrace()
      .UseReactiveUI();
}
