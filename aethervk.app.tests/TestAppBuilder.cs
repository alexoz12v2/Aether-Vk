using AetherVk.Utils;
using Avalonia;
using Avalonia.Headless;
using Microsoft.Extensions.Hosting;

[assembly: AvaloniaTestApplication(typeof(AetherVk.AppTests.TestAppBuilder))]

namespace AetherVk.AppTests;

public class TestAppBuilder
{
  public static AppBuilder BuildAvaloniaApp()
  {
    var host = Host.CreateDefaultBuilder()
      .ConfigureServices(
        (context, services) =>
        {
          services.AddCommonServices();
          services.AddViewModels();
        }
      )
      .Build();

    AetherVk.App.Host = host;
    host.Start();

    return AppBuilder.Configure<AetherVk.App>().UseHeadless(new AvaloniaHeadlessPlatformOptions());
  }
}
