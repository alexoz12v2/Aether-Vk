using System;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests;

public class HorizonJplServiceTests
{
  [Fact]
  public async Task FetchCometsAndDataTest()
  {
    var console = new ConsoleService();
    var breadcrumb = new BreadcrumbService();
    var service = new HorizonJplService(console, breadcrumb);

    await service.FetchCometsAsync("2000-01-01", "2020-01-01");
    Assert.NotEmpty(service.CometsData);

    var comets = service.CometsData.Take(5).ToList();

    foreach (var comet in comets)
    {
      var spkId = comet[3]; 
      
      await service.FetchDataAsync(
          command: $"DES={spkId}; CAP",
          startTime: "2024-01-01",
          stopTime: "2024-01-10",          stepSize: "1 d",
          center: "500@399"
      );
      
      Assert.NotEmpty(service.SessionData);
    }
  }
}
