// Linked into every .NET test project that reads test_assets/ (see each .csproj). One copy per
// assembly, so the types are internal.
#nullable enable
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Formats.Tar;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Security.Cryptography;
using System.Threading;

namespace AetherVk.TestAssets;

/// <summary>
/// The test assets listed in <c>test_assets/ASSETS.tsv</c>, downloaded on demand. A path relative
/// to test_assets/ resolves to the checkout's own file when it is there, otherwise to the group's
/// extracted tarball under the cache (<c>$AETHERVK_TEST_ASSETS_CACHE</c>, else
/// <c>&lt;repo&gt;/target/test-assets</c>), fetched from <c>$AETHERVK_TEST_ASSETS_MIRROR</c> or the
/// manifest's base URL (<c>$AETHERVK_TEST_ASSETS_BASE</c> overrides it), size and sha256 checked.
/// A test that cannot get its asset is skipped, or fails when <c>AETHERVK_TEST_ASSETS_REQUIRED=1</c>.
/// The Rust side (aethervk-test-assets) follows the same contract and locks the same files.
/// The cache is shared: a test that changes an asset works on a copy.
/// </summary>
internal static class TestAssets
{
  /// <summary>Where to look and fetch from; <see cref="FromEnvironment"/> is what the tests use.</summary>
  internal sealed record Options(string RepoRoot, string CacheRoot, string? Mirror, string? BaseOverride)
  {
    /// <summary>Called once per tarball extracted (tests count the fills with it).</summary>
    public Action<string>? OnExtract { get; init; }
  }

  internal sealed record Group(string Name, string[] Paths, string File, string Sha256, long Size);

  internal sealed record Manifest(string? Base, IReadOnlyList<Group> Groups);

  private static readonly TimeSpan ConnectTimeout = TimeSpan.FromSeconds(15);
  private static readonly TimeSpan TotalTimeout = TimeSpan.FromSeconds(300);

  /// <summary>True when <c>AETHERVK_TEST_ASSETS_REQUIRED=1</c> (CI): a missing asset fails the test.</summary>
  public static bool Required => Environment.GetEnvironmentVariable("AETHERVK_TEST_ASSETS_REQUIRED") == "1";

  /// <summary>The checkout root: the first folder above the test output holding test_assets/ASSETS.tsv.</summary>
  public static string? RepoRoot()
  {
    for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
      if (System.IO.File.Exists(Path.Combine(d.FullName, "test_assets", "ASSETS.tsv")))
        return d.FullName;
    return null;
  }

  /// <summary>The options from the environment, or null (with the reason) without a checkout.</summary>
  public static Options? FromEnvironment(out string reason)
  {
    var root = RepoRoot();
    if (root is null)
    {
      reason = $"test_assets/ASSETS.tsv not found above {AppContext.BaseDirectory}";
      return null;
    }
    reason = "";
    static string? Env(string name) => Environment.GetEnvironmentVariable(name) is { Length: > 0 } v ? v : null;
    return new Options(root, Env("AETHERVK_TEST_ASSETS_CACHE") ?? Path.Combine(root, "target", "test-assets"),
      Env("AETHERVK_TEST_ASSETS_MIRROR"), Env("AETHERVK_TEST_ASSETS_BASE"));
  }

  /// <summary>The asset's path (a file or a folder), fetching its group if needed; null with the reason when it cannot.</summary>
  public static string? TryPath(string rel, out string reason)
  {
    var options = FromEnvironment(out reason);
    return options is null ? null : TryPath(rel, options, out reason);
  }

  /// <summary>The asset's path; skips the test when it cannot be had, or fails it when assets are required.</summary>
  public static string Require(string rel)
  {
    var path = TryPath(rel, out var reason);
    if (path is not null)
      return path;
    var message = $"test asset {rel} unavailable: {reason}";
    if (Required)
      throw new InvalidOperationException(message + " (AETHERVK_TEST_ASSETS_REQUIRED=1)");
    Xunit.Skip.If(true, message);
    throw new InvalidOperationException(message); // not reached: Skip.If throws
  }

  public static string? TryPath(string rel, Options options, out string reason)
  {
    reason = "";
    rel = rel.Replace('\\', '/').Trim('/');
    var local = Path.Combine(options.RepoRoot, "test_assets", rel);
    if (System.IO.File.Exists(local) || Directory.Exists(local))
      return local;

    Manifest manifest;
    try
    {
      manifest = ReadManifest(Path.Combine(options.RepoRoot, "test_assets", "ASSETS.tsv"));
    }
    catch (Exception e) when (e is IOException or UnauthorizedAccessException or FormatException)
    {
      reason = $"cannot read ASSETS.tsv: {e.Message}";
      return null;
    }
    var group = FindGroup(manifest, rel);
    if (group is null)
    {
      reason = $"{rel} is not in ASSETS.tsv";
      return null;
    }
    try
    {
      var dir = EnsureGroup(manifest, group, options);
      return Path.Combine(dir, rel.Replace('/', Path.DirectorySeparatorChar));
    }
    catch (Exception e)
    {
      reason = $"group {group.Name} ({group.File}): {e.GetType().Name}: {e.Message}";
      return null;
    }
  }

  internal static Manifest ReadManifest(string path)
  {
    string? baseUrl = null;
    var groups = new List<Group>();
    foreach (var raw in System.IO.File.ReadAllLines(path))
    {
      var line = raw.TrimEnd('\r');
      if (line.Length == 0)
        continue;
      if (line.StartsWith('#'))
      {
        var body = line.TrimStart('#').Trim();
        if (body.StartsWith("base ", StringComparison.Ordinal))
          baseUrl = body.Substring(5).Trim();
        continue;
      }
      var cols = line.Split('\t');
      if (cols.Length != 5)
        throw new FormatException($"ASSETS.tsv row with {cols.Length} columns, not 5: {line}");
      var paths = cols[1].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
      for (int i = 0; i < paths.Length; i++)
        paths[i] = paths[i].Trim('/');
      groups.Add(new Group(cols[0].Trim(), paths, cols[2].Trim(), cols[3].Trim().ToLowerInvariant(),
        long.Parse(cols[4].Trim(), System.Globalization.CultureInfo.InvariantCulture)));
    }
    return new Manifest(baseUrl, groups);
  }

  /// <summary>The group with a member equal to <paramref name="rel"/> or a folder holding it.</summary>
  internal static Group? FindGroup(Manifest manifest, string rel)
  {
    foreach (var g in manifest.Groups)
      foreach (var e in g.Paths)
        if (rel == e || rel.StartsWith(e + "/", StringComparison.Ordinal))
          return g;
    return null;
  }

  internal static string GroupDir(Group group, Options options) =>
    Path.Combine(options.CacheRoot, $"{group.Name}-{group.Sha256[..12]}");

  /// <summary>The group's extracted folder, filled under the cross-process lock if not complete.</summary>
  internal static string EnsureGroup(Manifest manifest, Group group, Options options)
  {
    var dir = GroupDir(group, options);
    var complete = Path.Combine(dir, ".complete");
    if (System.IO.File.Exists(complete))
      return dir;
    Directory.CreateDirectory(options.CacheRoot);
    using var _ = Lock(Path.Combine(options.CacheRoot, group.Name + ".lock"));
    if (System.IO.File.Exists(complete))
      return dir;

    var pid = Environment.ProcessId;
    var tarball = Path.Combine(options.CacheRoot, $"{group.File}.download-{pid}");
    var tmp = $"{dir}.tmp-{pid}";
    try
    {
      Fetch(manifest, group, options, tarball);
      Verify(group, tarball);
      if (Directory.Exists(tmp))
        Directory.Delete(tmp, recursive: true);
      Directory.CreateDirectory(tmp);
      using (var file = System.IO.File.OpenRead(tarball))
      using (var gz = new GZipStream(file, CompressionMode.Decompress))
        TarFile.ExtractToDirectory(gz, tmp, overwriteFiles: false);
      options.OnExtract?.Invoke(group.Name);
      if (Directory.Exists(dir)) // a fill that died before .complete
        Directory.Delete(dir, recursive: true);
      Directory.Move(tmp, dir);
      System.IO.File.WriteAllText(complete, $"{group.File} {group.Sha256}\n");
      return dir;
    }
    finally
    {
      TryDelete(tarball);
      if (Directory.Exists(tmp))
        try { Directory.Delete(tmp, recursive: true); } catch (IOException) { }
    }
  }

  /// <summary>
  /// An exclusive lock on <paramref name="path"/>: FileShare.None is a flock on Unix and a sharing
  /// lock on Windows, so it excludes other processes, other threads and the Rust side alike.
  /// </summary>
  private static FileStream Lock(string path)
  {
    var deadline = Stopwatch.StartNew();
    while (true)
    {
      try
      {
        return new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
      }
      catch (IOException) when (deadline.Elapsed < TimeSpan.FromMinutes(15))
      {
        Thread.Sleep(50);
      }
    }
  }

  private static void Fetch(Manifest manifest, Group group, Options options, string target)
  {
    if (options.Mirror is { } mirror)
    {
      System.IO.File.Copy(Path.Combine(mirror, group.File), target, overwrite: true);
      return;
    }
    var baseUrl = options.BaseOverride ?? manifest.Base
      ?? throw new InvalidOperationException("ASSETS.tsv has no '# base' line and AETHERVK_TEST_ASSETS_BASE is unset");
    if (!baseUrl.EndsWith('/'))
      baseUrl += "/";
    var url = baseUrl + group.File;
    using var handler = new SocketsHttpHandler { ConnectTimeout = ConnectTimeout };
    using var http = new HttpClient(handler) { Timeout = TotalTimeout };
    using var cts = new CancellationTokenSource(TotalTimeout);
    using var response = http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cts.Token).GetAwaiter().GetResult();
    if (!response.IsSuccessStatusCode)
      throw new HttpRequestException($"GET {url}: {(int)response.StatusCode} {response.ReasonPhrase}");
    using var body = response.Content.ReadAsStream(cts.Token);
    using var file = new FileStream(target, FileMode.Create, FileAccess.Write, FileShare.None);
    body.CopyTo(file);
  }

  private static void Verify(Group group, string tarball)
  {
    var size = new FileInfo(tarball).Length;
    if (size != group.Size)
      throw new InvalidDataException($"{group.File} is {size} bytes, ASSETS.tsv says {group.Size}");
    using var f = System.IO.File.OpenRead(tarball);
    var sha = Convert.ToHexString(SHA256.HashData(f)).ToLowerInvariant();
    if (sha != group.Sha256)
      throw new InvalidDataException($"{group.File} sha256 {sha}, ASSETS.tsv says {group.Sha256}");
  }

  private static void TryDelete(string path)
  {
    try { System.IO.File.Delete(path); } catch (IOException) { } catch (UnauthorizedAccessException) { }
  }
}
