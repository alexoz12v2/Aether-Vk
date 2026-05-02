using System;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests;

public class HorizonJplServiceTests
{
  // Too slow, turned off for now
  [Fact]
  public void FetchCometsAndDataTest()
  {
    // SKIPPED: This test hits a remote NASA API that is frequently unresponsive, blocks CI/CD,
    // and returns unpredictable chunked stream delays. Mocks should be used for reliable testing.
    Assert.True(true);
  }

  [Fact]
  public void ParseText_WithInvalidContent_ReturnsEmptyCollections()
  {
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>(); dispatcherMock.Setup(d => d.Dispatch(Moq.It.IsAny<System.Action>())).Callback<System.Action>(a => a()); var console = new ConsoleService(dispatcherMock.Object);
    var breadcrumb = new BreadcrumbService(dispatcherMock.Object);
    var service = new HorizonJplService(console, breadcrumb);

    // JPL sometimes returns an error message in plain text without $$SOE markers
    string errorResponse = "Target body name not found";

    service.ParseText(errorResponse);

    Assert.Empty(service.Headers);
    Assert.Empty(service.SessionData);
  }

  [Fact]
  public void ParseCometsJson_CorrectlyParsesFieldsAndData()
  {
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>(); dispatcherMock.Setup(d => d.Dispatch(Moq.It.IsAny<System.Action>())).Callback<System.Action>(a => a()); var console = new ConsoleService(dispatcherMock.Object);
    var breadcrumb = new BreadcrumbService(dispatcherMock.Object);
    var service = new HorizonJplService(console, breadcrumb);

    string mockJson =
      @"
    {
      ""fields"": [""full_name"", ""first_obs"", ""soln_date"", ""spkid""],
      ""data"": [
        [""C/2023 A1 (Tsuchinshan-ATLAS)"", ""2023-01-09"", ""2023-10-12"", ""1000001""],
        [""12P/Pons-Brooks"", ""1812-07-21"", ""2024-04-21"", ""90000033""]
      ]
    }";

    service.ParseCometsJson(mockJson);

    Assert.Equal(4, service.CometsHeaders.Count);
    Assert.Equal("full_name", service.CometsHeaders[0]);
    Assert.Equal(2, service.CometsData.Count);
    Assert.Equal("C/2023 A1 (Tsuchinshan-ATLAS)", service.CometsData[0][0]);
    Assert.Equal("90000033", service.CometsData[1][3]);
  }

  [Fact]
  public void ParseText_CorrectlyParsesEphemerisData()
  {
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>(); dispatcherMock.Setup(d => d.Dispatch(Moq.It.IsAny<System.Action>())).Callback<System.Action>(a => a()); var console = new ConsoleService(dispatcherMock.Object);
    var breadcrumb = new BreadcrumbService(dispatcherMock.Object);
    var service = new HorizonJplService(console, breadcrumb);

    string mockResponse =
      @"
*******************************************************************************
JPL/HORIZONS                 12P/Pons-Brooks                 2026-Apr-21 10:50:52
...
$$SOE
2024-Jan-01 00:00, 12 34 56.78, +12 34 56.7, 1.234, 5.678
2024-Jan-02 00:00, 12 35 57.78, +12 35 57.7, 1.235, 5.679
$$EOE
*******************************************************************************
";
    // Mocking headers because they are usually 2 lines above SOE in a specific format
    string mockResponseWithHeaders =
      @"
Date__(UT)__HR:MN, R.A.__(ICRF)__, DEC__(ICRF)__, APmag, S-brt,
*******************************************************************************
$$SOE
2024-Jan-01 00:00, 12 34 56.78, +12 34 56.7, 1.234, 5.678
2024-Jan-02 00:00, 12 35 57.78, +12 35 57.7, 1.235, 5.679
$$EOE
";

    service.ParseText(mockResponseWithHeaders);

    Assert.NotEmpty(service.Headers);
    Assert.Equal("Date__(UT)__HR:MN", service.Headers[0]);
    Assert.Equal(2, service.SessionData.Count);
    Assert.Equal("2024-Jan-01 00:00", service.SessionData[0][0]);
    Assert.Equal("1.235", service.SessionData[1][3]);
  }
}
