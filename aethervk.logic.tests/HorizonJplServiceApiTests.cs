using System;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using Moq;
using Xunit;

namespace aethervk.logic.tests
{
  public class MockHttpMessageHandler : HttpMessageHandler
  {
    public string? LastRequestUrl { get; private set; }

    protected override Task<HttpResponseMessage> SendAsync(
      HttpRequestMessage request,
      CancellationToken cancellationToken
    )
    {
      LastRequestUrl = request.RequestUri?.ToString();
      return Task.FromResult(
        new HttpResponseMessage
        {
          StatusCode = HttpStatusCode.OK,
          Content = new StringContent("Mock response\n$$SOE\nA=1.0\nEC=0.5\nIN=0.1\nMA=0.2\n$$EOE"),
        }
      );
    }
  }

  public class HorizonJplServiceApiTests
  {
    private (HorizonJplService service, MockHttpMessageHandler handler) CreateService()
    {
      var handler = new MockHttpMessageHandler();
      var dispatcherMock = new Mock<IUiThreadDispatcher>();
      var console = new ConsoleService(dispatcherMock.Object);
      var breadcrumb = new BreadcrumbService();
      var storageMock = new Mock<ILocalStorageService>();

      // Setup storage mock to return some valid paths so it doesn't fail File operations
      storageMock.Setup(s => s.GetSessionPath(It.IsAny<string>())).Returns("dummy.txt");
      storageMock
        .Setup(s => s.GetPersistentPath(It.IsAny<string>()))
        .Returns("dummy_persistent.txt");

      var service = new HorizonJplService(console, breadcrumb, storageMock.Object);
      service._httpClient = new HttpClient(handler);
      return (service, handler);
    }

    [Fact]
    public async Task FetchObjectConstantsAsync_GeneratesCorrectUrl()
    {
      var (service, handler) = CreateService();

      // Expected equivalent of:
      // curl -G "https://ssd.jpl.nasa.gov/api/horizons.api" -d "format=text" --data-urlencode "COMMAND='90000030;'" --data-urlencode "MAKE_EPHEM='NO'" --data-urlencode "OBJ_DATA='YES'" --data-urlencode "CENTER='@10'"

      await service.FetchObjectConstantsAsync("90000030");

      Assert.NotNull(handler.LastRequestUrl);
      var url = handler.LastRequestUrl;

      Assert.Contains("format=text", url);
      Assert.Contains("COMMAND=%2790000030%3B%27", url); // URL-encoded '90000030;'
      Assert.Contains("MAKE_EPHEM=%27NO%27", url);
      Assert.Contains("OBJ_DATA=%27YES%27", url);
      Assert.Contains("CENTER=%27%4010%27", url); // URL-encoded '@10'
    }

    [Fact]
    public async Task FetchPlanetOrbitDataAsync_GeneratesCorrectUrl()
    {
      var (service, handler) = CreateService();

      // Expected equivalent of:
      // curl -G "https://ssd.jpl.nasa.gov/api/horizons.api" -d "format=text" --data-urlencode "COMMAND='90000033;'" --data-urlencode "MAKE_EPHEM='YES'" --data-urlencode "EPHEM_TYPE='ELEMENTS'" --data-urlencode "OBJ_DATA='NO'" --data-urlencode "CENTER='@10'" --data-urlencode "START_TIME='2026-05-27'" --data-urlencode "STOP_TIME='2026-06-27'" --data-urlencode "STEP_SIZE='1d'"

      await service.FetchPlanetOrbitDataAsync(
        "90000033",
        "@10",
        new DateTime(2026, 5, 27),
        new DateTime(2026, 6, 27),
        "1d"
      );

      Assert.NotNull(handler.LastRequestUrl);
      var url = handler.LastRequestUrl;

      Assert.Contains("format=text", url);
      Assert.Contains("COMMAND=%2790000033%3B%27", url);
      Assert.Contains("MAKE_EPHEM=%27YES%27", url);
      Assert.Contains("EPHEM_TYPE=%27ELEMENTS%27", url);
      Assert.Contains("OBJ_DATA=%27NO%27", url);
      Assert.Contains("CENTER=%27%4010%27", url);
      Assert.Contains("START_TIME=%272026-05-27%27", url);
      Assert.Contains("STOP_TIME=%272026-06-27%27", url);
      Assert.Contains("STEP_SIZE=%271d%27", url);
    }

    [Fact]
    public async Task DownloadSpkByIdAsync_GeneratesCorrectUrl()
    {
      var (service, handler) = CreateService();

      // Expected equivalent of Enumerate Descriptors (but for SPK):
      // curl -G "https://ssd.jpl.nasa.gov/api/horizons.api" -d "format=text" --data-urlencode "COMMAND='90000033;'" --data-urlencode "MAKE_EPHEM='YES'" --data-urlencode "OBJ_DATA='NO'" --data-urlencode "START_TIME='2026-05-27'" --data-urlencode "STOP_TIME='2026-06-27'" --data-urlencode "EPHEM_TYPE='SPK'"

      await service.DownloadSpkByIdAsync(
        "pdes",
        "90000033",
        "dummy.bsp",
        "2026-05-27",
        "2026-06-27"
      );

      Assert.NotNull(handler.LastRequestUrl);
      var url = handler.LastRequestUrl;

      Assert.Contains("format=text", url);
      Assert.Contains("COMMAND=%2790000033%3B%27", url);
      Assert.Contains("MAKE_EPHEM=%27YES%27", url);
      Assert.Contains("EPHEM_TYPE=%27SPK%27", url);
      Assert.Contains("OBJ_DATA=%27NO%27", url);
      // We expect 2026-05-26 and 2026-06-29 because the JPL Horizons STOP_TIME is exclusive,
      // and we aggressively pad both bounds to prevent microsecond floating point truncation
      // panics when the simulation evaluates exactly at the limits.
      Assert.Contains("START_TIME=%272026-05-26%27", url);
      Assert.Contains("STOP_TIME=%272026-06-29%27", url);
    }
  
    private sealed class FixedResponseHandler : HttpMessageHandler
    {
      private readonly string _body;
      public string? LastRequestUrl { get; private set; }

      public FixedResponseHandler(string body) => _body = body;

      protected override Task<HttpResponseMessage> SendAsync(
        HttpRequestMessage request,
        CancellationToken cancellationToken
      )
      {
        LastRequestUrl = request.RequestUri?.ToString();
        return Task.FromResult(
          new HttpResponseMessage { StatusCode = HttpStatusCode.OK, Content = new StringContent(_body) }
        );
      }
    }

    /// The post-commit nucleus radius is fetched by apparition record id: a bare designation
    /// ("67P;") makes Horizons answer with the record index, which used to raise a spurious
    /// error breadcrumb after a successful commit.
    [Fact]
    public async Task FetchObjectConstantsAsync_ByRecordId_ParsesCometRadius()
    {
      var handler = new FixedResponseHandler(
        "Rec #:90000703 (+COV) Soln.date: 2026-Sep-04_10:25:07\n"
          + "Comet physical (GM= km^3/s^2; RAD= km):\n"
          + "   GM= 6.622E-7            RAD= 1.7\n"
      );
      var storageMock = new Mock<ILocalStorageService>();
      storageMock
        .Setup(s => s.GetPersistentPath(It.IsAny<string>()))
        .Returns(System.IO.Path.Combine(System.IO.Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".txt"));
      var breadcrumb = new BreadcrumbService();
      var service = new HorizonJplService(
        new ConsoleService(new Mock<IUiThreadDispatcher>().Object),
        breadcrumb,
        storageMock.Object
      );
      service._httpClient = new HttpClient(handler);

      var (radiusKm, massKg) = await service.FetchObjectConstantsAsync("90000703");

      Assert.Contains("COMMAND=%2790000703%3B%27", handler.LastRequestUrl);
      Assert.Equal(1.7, radiusKm, 6);
      Assert.Equal(6.622e-7 / 6.6743e-20, massKg, 1);
    }
}
}
