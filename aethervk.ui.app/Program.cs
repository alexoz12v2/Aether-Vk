using System;
using AetherVk.Utils;
using Avalonia;
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
        // 1. Setup Microsoft Hosting
        var host = Host.CreateDefaultBuilder(args)
            .ConfigureServices(
                (context, services) =>
                {
                    services.AddCommonServices();
                    services.AddViewModels();
                    // Register custom services (todo)
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
    // TODO: ReactiveUI.Avalonia;
    public static AppBuilder BuildAvaloniaApp() =>
        AppBuilder.Configure<App>().UsePlatformDetect().WithInterFont().LogToTrace();
}
