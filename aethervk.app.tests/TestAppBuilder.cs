using Avalonia;
using Avalonia.Headless;
using Microsoft.Extensions.Hosting;
using AetherVk.Utils;

[assembly: AvaloniaTestApplication(typeof(AetherVk.App.Tests.TestAppBuilder))]

namespace AetherVk.App.Tests;

public class TestAppBuilder
{
    public static AppBuilder BuildAvaloniaApp()
    {
        var host = Host.CreateDefaultBuilder()
          .ConfigureServices((context, services) =>
          {
              services.AddCommonServices();
              services.AddViewModels();
          })
          .Build();
          
        AetherVk.App.Host = host;
        host.Start();

        return AppBuilder.Configure<AetherVk.App>()
            .UseHeadless(new AvaloniaHeadlessPlatformOptions());
    }
}
