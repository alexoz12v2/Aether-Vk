using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// NAIF leap-second kernel updates: listing parse, cache skip, size cap, rejected files, offline,
/// cache-before-network. Fake HTTP handler and temp folders only; no network.
/// </summary>
public class LeapSecondsUpdateServiceTests : IDisposable
{
  private const string ListingWithNoise = """
    <html><head><title>Index of /pub/naif/generic_kernels/lsk</title></head><body>
    <a href="?C=N;O=D">Name</a> <a href="/pub/naif/generic_kernels/">Parent Directory</a>
    <a href="a_old_versions/">a_old_versions/</a>
    <a href="aareadme.txt">aareadme.txt</a>
    <a href="latest_leapseconds.tls">latest_leapseconds.tls</a>
    <a href="naif0011.tls">naif0011.tls</a>
    <a href="naif0012.tls">naif0012.tls</a>
    <a href="naif0012.tls.pc">naif0012.tls.pc</a>
    <a href="naif0099.tls.lbl">naif0099.tls.lbl</a>
    <a href="xnaif0098.tls">xnaif0098.tls</a>
    <a href="naif0013.tlsx">naif0013.tlsx</a>
    <a href="naif0013.tls">naif0013.tls</a>
    <a href="naif00100.tls">naif00100.tls</a>
    </body></html>
    """;

  private static readonly byte[] KernelBytes = Encoding.ASCII.GetBytes("KPL/LSK\n\\begindata\nDELTET/DELTA_AT = ( 10, @1972-JAN-1 )\n\\begintext\n");

  private readonly string _dir = Path.Combine(Path.GetTempPath(), "aethervk_lsk_" + Guid.NewGuid().ToString("N"));

  public void Dispose()
  {
    try
    {
      if (Directory.Exists(_dir))
        Directory.Delete(_dir, recursive: true);
    }
    catch (IOException) { }
  }

  private sealed class FakeHandler : HttpMessageHandler
  {
    public readonly Dictionary<string, Func<HttpResponseMessage>> Routes = new();
    public readonly List<string> Requested = new();
    public bool Offline;

    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
      var url = request.RequestUri!.ToString();
      Requested.Add(url);
      if (Offline)
        throw new HttpRequestException("network unreachable");
      return Task.FromResult(Routes.TryGetValue(url, out var make) ? make() : new HttpResponseMessage(HttpStatusCode.NotFound));
    }
  }

  private sealed class Loader
  {
    public readonly List<string> Paths = new();
    public readonly List<string> Contents = new();
    public int Status;
    public uint Entries = 28;

    public (int, uint) Load(string path)
    {
      Paths.Add(path);
      Contents.Add(File.Exists(path) ? File.ReadAllText(path) : "<missing>");
      return (Status, Entries);
    }
  }

  private (LeapSecondsUpdateService svc, FakeHandler http, Loader loader, List<string> log) Make()
  {
    var http = new FakeHandler();
    http.Routes[LeapSecondsUpdateService.ListingUrl] = () => new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(ListingWithNoise) };
    http.Routes[LeapSecondsUpdateService.ListingUrl + "naif0013.tls"] = () => new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(KernelBytes) };
    var loader = new Loader();
    var log = new List<string>();
    var svc = new LeapSecondsUpdateService(new HttpClient(http), _dir, loader.Load, log.Add, TimeSpan.FromSeconds(5));
    return (svc, http, loader, log);
  }

  private void Cache(string name, string text = "cached")
  {
    Directory.CreateDirectory(_dir);
    File.WriteAllText(Path.Combine(_dir, name), text);
  }

  [Fact]
  public void Listing_PicksHighestNaif00NN_IgnoringNoiseLinks()
  {
    Assert.Equal(13, LeapSecondsUpdateService.ParseHighestKernelNumber(ListingWithNoise));
    Assert.Null(LeapSecondsUpdateService.ParseHighestKernelNumber("<a href=\"latest_leapseconds.tls\">x</a>"));
    Assert.Null(LeapSecondsUpdateService.ParseHighestKernelNumber(null));
    Assert.Equal("naif0013.tls", LeapSecondsUpdateService.KernelFileName(13));
  }

  [Fact]
  public async Task Update_DownloadsHighest_MovesIntoCache_LoadsIt_RaisesEvent()
  {
    var (svc, http, loader, _) = Make();
    LeapSecondsSourceChangedEventArgs? raised = null;
    svc.LeapSecondsSourceChanged += (_, e) => raised = e;

    var status = await svc.UpdateAsync();

    Assert.Equal(LeapSecondsLoadStatus.Adopted, status);
    var final = Path.Combine(_dir, "naif0013.tls");
    Assert.Equal(new[] { final }, loader.Paths);
    Assert.Equal(Encoding.ASCII.GetString(KernelBytes), loader.Contents[0]);
    Assert.False(File.Exists(final + ".part"));
    Assert.Contains(LeapSecondsUpdateService.ListingUrl + "naif0013.tls", http.Requested);
    Assert.NotNull(raised);
    Assert.Equal(13, raised!.KernelNumber);
    Assert.True(raised.Downloaded);
    Assert.Equal(28u, raised.Entries);
    Assert.Same(raised, svc.CurrentSource);
  }

  [Fact]
  public async Task Update_SkipsWhenCachedIsCurrent()
  {
    var (svc, http, loader, _) = Make();
    Cache("naif0013.tls");

    Assert.Null(await svc.UpdateAsync());
    Assert.Empty(loader.Paths);
    Assert.DoesNotContain(LeapSecondsUpdateService.ListingUrl + "naif0013.tls", http.Requested);
  }

  [Fact]
  public async Task Update_OversizedResponseRejected_NothingCachedOrLoaded()
  {
    var (svc, http, loader, log) = Make();
    var big = new byte[LeapSecondsUpdateService.MaxKernelBytes + 1];
    // no Content-Length either way: a streamed body must be cut too
    http.Routes[LeapSecondsUpdateService.ListingUrl + "naif0013.tls"] =
      () => new HttpResponseMessage(HttpStatusCode.OK) { Content = new StreamContent(new NonSeekableStream(big)) };

    Assert.Null(await svc.UpdateAsync());
    Assert.Empty(loader.Paths);
    Assert.False(File.Exists(Path.Combine(_dir, "naif0013.tls")));
    Assert.False(File.Exists(Path.Combine(_dir, "naif0013.tls.part")));
    Assert.Single(log);

    // with a declared Content-Length the cap applies before reading
    var (svc2, http2, loader2, _) = Make();
    http2.Routes[LeapSecondsUpdateService.ListingUrl + "naif0013.tls"] =
      () => new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(big) };
    Assert.Null(await svc2.UpdateAsync());
    Assert.Empty(loader2.Paths);
  }

  [Fact]
  public async Task LoaderInconsistent_FileRenamedRejected_NotReloadedNorRedownloadedNextStart()
  {
    var (svc, _, loader, _) = Make();
    loader.Status = (int)LeapSecondsLoadStatus.Inconsistent;
    LeapSecondsSourceChangedEventArgs? raised = null;
    svc.LeapSecondsSourceChanged += (_, e) => raised = e;

    Assert.Equal(LeapSecondsLoadStatus.Inconsistent, await svc.UpdateAsync());
    Assert.False(File.Exists(Path.Combine(_dir, "naif0013.tls")));
    Assert.True(File.Exists(Path.Combine(_dir, "naif0013.tls" + LeapSecondsUpdateService.RejectedSuffix)));
    Assert.Null(raised);
    Assert.Null(svc.CurrentSource);

    // next start: the cache holds nothing loadable, and the listing's naif0013 is not fetched again
    var (next, http2, loader2, _) = Make();
    Assert.Null(next.LoadCachedKernel());
    Assert.Null(await next.UpdateAsync());
    Assert.Empty(loader2.Paths);
    Assert.DoesNotContain(LeapSecondsUpdateService.ListingUrl + "naif0013.tls", http2.Requested);
  }

  [Fact]
  public async Task Offline_NoThrow_LoaderNotCalled_LoggedOnce()
  {
    var (svc, http, loader, log) = Make();
    http.Offline = true;

    Assert.Null(await svc.UpdateAsync());
    Assert.Null(await svc.UpdateAsync());
    await svc.StartBackgroundUpdate();
    Assert.Empty(loader.Paths);
    Assert.Single(log);
    Assert.False(Directory.Exists(_dir) && Directory.GetFiles(_dir).Length > 0);
  }

  [Fact]
  public async Task DownloadDroppedMidBody_NoPartialFileLoaded()
  {
    var (svc, http, loader, _) = Make();
    http.Routes[LeapSecondsUpdateService.ListingUrl + "naif0013.tls"] =
      () => new HttpResponseMessage(HttpStatusCode.OK) { Content = new StreamContent(new FailingStream(KernelBytes, failAfter: 10)) };

    Assert.Null(await svc.UpdateAsync());
    Assert.Empty(loader.Paths);
    Assert.False(File.Exists(Path.Combine(_dir, "naif0013.tls")));
    Assert.False(File.Exists(Path.Combine(_dir, "naif0013.tls.part")));
  }

  [Fact]
  public void CachedKernelLoadedBeforeNetwork_NewestFirst_RejectedFallsBack()
  {
    var (svc, http, loader, _) = Make();
    Cache("naif0011.tls", "eleven");
    Cache("naif0012.tls", "twelve");
    Cache("naif0013.tls.rejected", "bad");
    Cache("naif0014.tls.part", "partial");

    Assert.Equal(LeapSecondsLoadStatus.Adopted, svc.LoadCachedKernel());
    Assert.Empty(http.Requested); // no network
    Assert.Equal(new[] { "twelve" }, loader.Contents);
    Assert.False(svc.CurrentSource!.Downloaded);
    Assert.Equal(12, svc.CurrentSource.KernelNumber);

    // a cached file the parser rejects is moved aside and the next older one is tried
    var contents = new List<string>();
    var svc2 = new LeapSecondsUpdateService(new HttpClient(new FakeHandler()), _dir, p =>
    {
      contents.Add(File.ReadAllText(p));
      return contents.Count == 1 ? ((int)LeapSecondsLoadStatus.Malformed, 0u) : (0, 27u);
    });
    Assert.Equal(LeapSecondsLoadStatus.Adopted, svc2.LoadCachedKernel());
    Assert.Equal(new[] { "twelve", "eleven" }, contents);
    Assert.True(File.Exists(Path.Combine(_dir, "naif0012.tls" + LeapSecondsUpdateService.RejectedSuffix)));
    Assert.False(File.Exists(Path.Combine(_dir, "naif0012.tls")));
  }

  [Fact]
  public async Task Cancelled_NoThrow_NothingLoaded()
  {
    var (svc, _, loader, log) = Make();
    using var cts = new CancellationTokenSource();
    cts.Cancel();
    Assert.Null(await svc.UpdateAsync(cts.Token));
    Assert.Empty(loader.Paths);
    Assert.Empty(log);
  }

  private sealed class NonSeekableStream(byte[] data) : MemoryStream(data)
  {
    public override bool CanSeek => false;
    public override long Length => throw new NotSupportedException();
  }

  private sealed class FailingStream(byte[] data, int failAfter) : Stream
  {
    private int _pos;
    public override bool CanRead => true;
    public override bool CanSeek => false;
    public override bool CanWrite => false;
    public override long Length => throw new NotSupportedException();
    public override long Position { get => _pos; set => throw new NotSupportedException(); }
    public override void Flush() { }
    public override int Read(byte[] buffer, int offset, int count)
    {
      if (_pos >= failAfter)
        throw new IOException("connection reset");
      var n = Math.Min(count, Math.Min(failAfter, data.Length) - _pos);
      Array.Copy(data, _pos, buffer, offset, n);
      _pos += n;
      return n;
    }
    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
    public override void SetLength(long value) => throw new NotSupportedException();
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
  }
}
