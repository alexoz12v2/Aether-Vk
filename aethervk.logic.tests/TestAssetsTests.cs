using System;
using System.Formats.Tar;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;
using Xunit;
using Assets = AetherVk.TestAssets.TestAssets;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The on-demand test assets (tests-shared/TestAssets.cs, test_assets/ASSETS.tsv): a checkout file
/// wins over the cache, a member resolves by exact name or as a folder prefix, a group is fetched
/// from the mirror with its size and sha256 checked, a wrong hash leaves no cache entry, and two
/// fills racing for one group extract it once. Built on tiny tarballs in a temp folder.
/// </summary>
public sealed class TestAssetsTests : IDisposable
{
  private readonly string _root = Path.Combine(Path.GetTempPath(), "avk_testassets_" + Guid.NewGuid().ToString("N"));
  private string Repo => Path.Combine(_root, "repo");
  private string Cache => Path.Combine(_root, "cache");
  private string Mirror => Path.Combine(_root, "mirror");

  public TestAssetsTests()
  {
    Directory.CreateDirectory(Path.Combine(Repo, "test_assets"));
    Directory.CreateDirectory(Mirror);
  }

  public void Dispose()
  {
    try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
  }

  /// <summary>A gzipped GNU tar of <paramref name="files"/> (test_assets-relative name → text) in the mirror; its sha256 and size.</summary>
  private (string Sha, long Size) Tarball(string name, params (string Path, string Text)[] files)
  {
    var path = Path.Combine(Mirror, name);
    using (var f = File.Create(path))
    using (var gz = new GZipStream(f, CompressionLevel.Optimal))
    using (var tar = new TarWriter(gz, TarEntryFormat.Gnu))
      foreach (var (p, text) in files)
      {
        var entry = new GnuTarEntry(TarEntryType.RegularFile, p) { DataStream = new MemoryStream(System.Text.Encoding.UTF8.GetBytes(text)) };
        tar.WriteEntry(entry);
      }
    var bytes = File.ReadAllBytes(path);
    return (Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant(), bytes.Length);
  }

  private void Manifest(params string[] rows) =>
    File.WriteAllText(Path.Combine(Repo, "test_assets", "ASSETS.tsv"),
      "# test manifest\n# base http://127.0.0.1:9/\n# group\tpaths\tfile\tsha256\tsize\n" + string.Join("\n", rows) + "\n");

  private static string Row(string group, string paths, string file, (string Sha, long Size) t) =>
    $"{group}\t{paths}\t{file}\t{t.Sha}\t{t.Size}";

  private Assets.Options Options(Action<string>? onExtract = null) =>
    new(Repo, Cache, Mirror, null) { OnExtract = onExtract };

  [Fact]
  public void ACheckoutFile_WinsOverTheCache()
  {
    var t = Tarball("g.tar.gz", ("a.txt", "from the tarball"));
    Manifest(Row("g", "a.txt", "g.tar.gz", t));
    File.WriteAllText(Path.Combine(Repo, "test_assets", "a.txt"), "from the checkout");

    var path = Assets.TryPath("a.txt", Options(), out var reason);
    Assert.Equal(Path.Combine(Repo, "test_assets", "a.txt"), path);
    Assert.Equal("", reason);
    Assert.False(Directory.Exists(Cache), "nothing fetched");

    File.Delete(Path.Combine(Repo, "test_assets", "a.txt"));
    path = Assets.TryPath("a.txt", Options(), out reason);
    Assert.NotNull(path);
    Assert.Equal(Path.Combine(Cache, "g-" + t.Sha[..12], "a.txt"), path);
    Assert.Equal("from the tarball", File.ReadAllText(path!));
    Assert.True(File.Exists(Path.Combine(Cache, "g-" + t.Sha[..12], ".complete")));
  }

  [Fact]
  public void AMember_MatchesExactly_OrAsAFolderPrefix_NotAsAStringPrefix()
  {
    var t = Tarball("saves.tar.gz", ("saves/s1/index.json", "{}"), ("saves/s1/spk/k.bsp", "kernel"), ("saves/s1.avk", "zip"));
    Manifest(Row("saves", "saves/s1,saves/s1.avk", "saves.tar.gz", t));

    var dir = Assets.TryPath("saves/s1", Options(), out _);
    Assert.True(Directory.Exists(dir));
    Assert.Equal("kernel", File.ReadAllText(Path.Combine(dir!, "spk", "k.bsp")));
    Assert.Equal("kernel", File.ReadAllText(Assets.TryPath("saves/s1/spk/k.bsp", Options(), out _)!));
    Assert.Equal("zip", File.ReadAllText(Assets.TryPath("saves/s1.avk", Options(), out _)!));

    Assert.Null(Assets.TryPath("saves/s10", Options(), out var reason));
    Assert.Contains("not in ASSETS.tsv", reason);
    Assert.Null(Assets.TryPath("saves", Options(), out reason));
    Assert.Contains("not in ASSETS.tsv", reason);
  }

  [Fact]
  public void TheMirror_IsUsed_InsteadOfTheBase()
  {
    var t = Tarball("m.tar.gz", ("m/x.txt", "mirrored"));
    Manifest(Row("m", "m", "m.tar.gz", t));
    // the manifest base (port 9) refuses connections: only the mirror can fill the group
    var path = Assets.TryPath("m/x.txt", Options(), out var reason);
    Assert.True(path is not null, reason);
    Assert.Equal("mirrored", File.ReadAllText(path!));

    // without the mirror the base is tried, and a refused connection is a reason, not an exception
    var noMirror = new Assets.Options(Repo, Path.Combine(_root, "cache2"), null, "http://127.0.0.1:9/");
    Assert.Null(Assets.TryPath("m/x.txt", noMirror, out reason));
    Assert.Contains("m.tar.gz", reason);
  }

  [Fact]
  public void AHashMismatch_FailsWithTheReason_AndLeavesNoGroupFolder()
  {
    var t = Tarball("h.tar.gz", ("h.txt", "payload"));
    Manifest(Row("h", "h.txt", "h.tar.gz", (new string('0', 64), t.Size)));
    Assert.Null(Assets.TryPath("h.txt", Options(), out var reason));
    Assert.Contains("sha256", reason);
    Assert.Empty(Directory.GetDirectories(Cache));
    Assert.Equal(["h.lock"], Directory.GetFiles(Cache).Select(Path.GetFileName).ToArray());

    Manifest(Row("h", "h.txt", "h.tar.gz", (t.Sha, t.Size + 1)));
    Assert.Null(Assets.TryPath("h.txt", Options(), out reason));
    Assert.Contains("bytes", reason);
  }

  [Fact]
  public void TwoConcurrentFills_ExtractOnce()
  {
    var files = Enumerable.Range(0, 200).Select(i => ($"c/f{i:D3}.txt", new string('x', 4096) + i)).ToArray();
    var t = Tarball("c.tar.gz", files);
    Manifest(Row("c", "c", "c.tar.gz", t));
    int extractions = 0;
    var options = Options(_ => Interlocked.Increment(ref extractions));

    using var start = new Barrier(4);
    var results = Enumerable.Range(0, 4).Select(i => Task.Factory.StartNew(() =>
    {
      start.SignalAndWait();
      return Assets.TryPath($"c/f{i:D3}.txt", options, out var reason) ?? throw new Xunit.Sdk.XunitException(reason);
    }, TaskCreationOptions.LongRunning)).ToArray();
    Task.WaitAll(results);

    Assert.Equal(1, extractions);
    for (int i = 0; i < 4; i++)
      Assert.Equal(new string('x', 4096) + i, File.ReadAllText(results[i].Result));
    Assert.Equal(200, Directory.GetFiles(Path.Combine(Cache, "c-" + t.Sha[..12], "c")).Length);
    Assert.DoesNotContain(Directory.GetDirectories(Cache), d => d.Contains(".tmp-"));
  }
}
