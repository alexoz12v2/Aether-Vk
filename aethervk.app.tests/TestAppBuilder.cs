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
    // as Program.Main: dot numbers on every OS locale (an it_IT Mac otherwise shows "∠ 0,2°")
    Logic.Utils.NumberCulture.Apply();

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

    return AppBuilder.Configure<AetherVk.App>()
      .UseSkia()
      .UseHeadless(new AvaloniaHeadlessPlatformOptions { UseHeadlessDrawing = false });
  }
}
