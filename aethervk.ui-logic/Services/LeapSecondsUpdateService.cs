using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace AetherVk.Logic.Services;

// LeapSecondsLoadStatus is defined next to the time-scale converter (ITimeScaleConverter.cs).

/// <summary>Raised after a leap-second kernel was handed to the native table.</summary>
public sealed class LeapSecondsSourceChangedEventArgs : EventArgs
{
  public LeapSecondsSourceChangedEventArgs(string path, int kernelNumber, LeapSecondsLoadStatus status, uint entries, bool downloaded)
  {
    Path = path;
    KernelNumber = kernelNumber;
    Status = status;
    Entries = entries;
    Downloaded = downloaded;
  }

  /// <summary>Cached kernel path (naif00NN.tls).</summary>
  public string Path { get; }

  /// <summary>NN of naif00NN.tls.</summary>
  public int KernelNumber { get; }

  public LeapSecondsLoadStatus Status { get; }

  /// <summary>Leap-second entries reported by the parser.</summary>
  public uint Entries { get; }

  /// <summary>True when the file was fetched in this run, false when it came from the cache.</summary>
  public bool Downloaded { get; }
}

/// <summary>
/// Keeps the native leap-second table current from NAIF's generic LSK directory.
/// </summary>
/// <remarks>
/// <para>Startup: <see cref="LoadCachedKernel"/> hands the newest cached naif00NN.tls to the native
/// parser before any network; <see cref="StartBackgroundUpdate"/> then (fire-and-forget, short
/// timeout, cancellable) reads the directory listing, picks the highest naif00NN.tls by NN and, when
/// it is newer than the cached one, downloads it (≤ <see cref="MaxKernelBytes"/>) to a temp file,
/// moves it into the cache and loads it.</para>
/// <para>The kernel is untrusted text: it is only stored and handed to the native parser, which
/// validates it. A file the parser rejects (Unreadable / Malformed / Inconsistent) is renamed to
/// <c>.rejected</c>, so it is neither loaded at the next start nor downloaded again. Offline or any
/// failure is logged once and the current table is kept.</para>
/// </remarks>
public sealed class LeapSecondsUpdateService
{
  public const string ListingUrl = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/";
  public const long MaxKernelBytes = 64 * 1024;
  public const string RejectedSuffix = ".rejected";
  public static readonly TimeSpan DefaultTimeout = TimeSpan.FromSeconds(20);

  // only these names are ever considered; everything else in the listing is ignored
  private static readonly Regex KernelName = new(@"(?<![\w.])naif00(\d\d)\.tls(?![\w.])", RegexOptions.CultureInvariant);
  private static readonly Regex CachedName = new(@"^naif00(\d\d)\.tls$", RegexOptions.CultureInvariant);

  private readonly HttpClient _http;
  private readonly string _cacheDirectory;
  private readonly Func<string, (int status, uint entries)> _loadLeapSecondsFile;
  private readonly Action<string>? _log;
  private readonly TimeSpan _timeout;
  private int _failureLogged;

  public LeapSecondsUpdateService(
    HttpClient http,
    string cacheDirectory,
    Func<string, (int status, uint entries)> loadLeapSecondsFile,
    Action<string>? log = null,
    TimeSpan? timeout = null
  )
  {
    _http = http ?? throw new ArgumentNullException(nameof(http));
    _cacheDirectory = cacheDirectory ?? throw new ArgumentNullException(nameof(cacheDirectory));
    _loadLeapSecondsFile = loadLeapSecondsFile ?? throw new ArgumentNullException(nameof(loadLeapSecondsFile));
    _log = log;
    _timeout = timeout ?? DefaultTimeout;
  }

  /// <summary>Default HttpClient (no client timeout; the per-run timeout is a cancellation).</summary>
  public static HttpClient CreateDefaultHttpClient()
  {
    var client = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
    client.DefaultRequestHeaders.Add("User-Agent", "AetherVk/1.0");
    return client;
  }

  /// <summary>Default cache folder: &lt;persistent&gt;/lsk (kept across sessions).</summary>
  public static string DefaultCacheDirectory(ILocalStorageService storage) => storage.GetPersistentPath("lsk");

  /// <summary>Raised (on the calling/background thread) after a kernel was handed to the native table.</summary>
  public event EventHandler<LeapSecondsSourceChangedEventArgs>? LeapSecondsSourceChanged;

  /// <summary>Last kernel handed to the parser and accepted (Adopted / KeptCurrent), null before.</summary>
  public LeapSecondsSourceChangedEventArgs? CurrentSource { get; private set; }

  public string CacheDirectory => _cacheDirectory;

  // ─────────────────────────────────────────────────────── Listing

  /// <summary>Highest NN among naif00NN.tls names in a directory listing, or null.</summary>
  public static int? ParseHighestKernelNumber(string? listingHtml)
  {
    if (string.IsNullOrEmpty(listingHtml))
      return null;
    int? best = null;
    foreach (Match m in KernelName.Matches(listingHtml))
    {
      var nn = int.Parse(m.Groups[1].Value, NumberStyles.None, CultureInfo.InvariantCulture);
      if (best is null || nn > best)
        best = nn;
    }
    return best;
  }

  public static string KernelFileName(int nn) => "naif00" + nn.ToString("00", CultureInfo.InvariantCulture) + ".tls";

  /// <summary>Highest NN of an accepted kernel in the cache (rejected / partial files ignored).</summary>
  public int? HighestCachedKernelNumber()
  {
    if (!Directory.Exists(_cacheDirectory))
      return null;
    int? best = null;
    foreach (var path in Directory.EnumerateFiles(_cacheDirectory))
    {
      var m = CachedName.Match(Path.GetFileName(path));
      if (!m.Success)
        continue;
      var nn = int.Parse(m.Groups[1].Value, NumberStyles.None, CultureInfo.InvariantCulture);
      if (best is null || nn > best)
        best = nn;
    }
    return best;
  }

  private bool IsRejected(int nn) => File.Exists(Path.Combine(_cacheDirectory, KernelFileName(nn) + RejectedSuffix));

  // ─────────────────────────────────────────────────────── Startup

  /// <summary>
  /// Loads the newest cached kernel (no network). A cached file the parser rejects is renamed to
  /// .rejected and the next older one is tried. Returns the status of the kernel that stayed, or null.
  /// </summary>
  public LeapSecondsLoadStatus? LoadCachedKernel()
  {
    int? previous = null;
    while (true)
    {
      int? nn;
      try
      {
        nn = HighestCachedKernelNumber();
      }
      catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
      {
        LogFailureOnce($"leap seconds: cache unreadable ({ex.Message}); keeping the built-in table");
        return null;
      }
      if (nn is null || nn == previous)
        return null; // nothing cached, or a rejected file could not be moved aside
      previous = nn;
      var status = LoadIntoNative(Path.Combine(_cacheDirectory, KernelFileName(nn.Value)), nn.Value, downloaded: false);
      if (status is null or LeapSecondsLoadStatus.Adopted or LeapSecondsLoadStatus.KeptCurrent)
        return status;
      // rejected → renamed; try the next older cached kernel
    }
  }

  /// <summary>Fire-and-forget startup update; never throws, never blocks the caller.</summary>
  public Task StartBackgroundUpdate(CancellationToken cancellationToken = default) =>
    Task.Run(() => UpdateAsync(cancellationToken), CancellationToken.None);

  /// <summary>
  /// Checks the NAIF listing and downloads + loads a newer kernel. Returns the status of the
  /// downloaded kernel, or null when nothing was downloaded (current, offline, failure, cancelled).
  /// Never throws.
  /// </summary>
  public async Task<LeapSecondsLoadStatus?> UpdateAsync(CancellationToken cancellationToken = default)
  {
    using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
    cts.CancelAfter(_timeout);
    string? tempPath = null;
    try
    {
      string listing;
      using (var response = await _http.GetAsync(ListingUrl, HttpCompletionOption.ResponseHeadersRead, cts.Token).ConfigureAwait(false))
      {
        response.EnsureSuccessStatusCode();
        listing = await ReadCappedStringAsync(response, 4 * 1024 * 1024, cts.Token).ConfigureAwait(false);
      }

      var remote = ParseHighestKernelNumber(listing);
      if (remote is null)
      {
        LogFailureOnce("leap seconds: no naif00NN.tls in the NAIF listing; keeping the current table");
        return null;
      }
      var cached = HighestCachedKernelNumber();
      if (cached is not null && cached >= remote)
        return null; // current
      if (IsRejected(remote.Value))
        return null; // rejected before: do not download or load it again

      Directory.CreateDirectory(_cacheDirectory);
      var name = KernelFileName(remote.Value);
      tempPath = Path.Combine(_cacheDirectory, name + ".part");
      using (var response = await _http.GetAsync(ListingUrl + name, HttpCompletionOption.ResponseHeadersRead, cts.Token).ConfigureAwait(false))
      {
        response.EnsureSuccessStatusCode();
        if (response.Content.Headers.ContentLength is long declared && declared > MaxKernelBytes)
          throw new InvalidDataException($"{name} is {declared} bytes (cap {MaxKernelBytes})");
        using var source = await response.Content.ReadAsStreamAsync().ConfigureAwait(false);
        using var target = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None);
        await CopyCappedAsync(source, target, MaxKernelBytes, name, cts.Token).ConfigureAwait(false);
      }

      var finalPath = Path.Combine(_cacheDirectory, name);
      if (File.Exists(finalPath))
        File.Delete(finalPath);
      File.Move(tempPath, finalPath);
      tempPath = null;
      _log?.Invoke($"leap seconds: downloaded {name}");
      return LoadIntoNative(finalPath, remote.Value, downloaded: true);
    }
    catch (Exception ex)
    {
      if (cancellationToken.IsCancellationRequested)
        return null;
      LogFailureOnce(
        ex is OperationCanceledException
          ? "leap seconds: NAIF did not answer in time; keeping the current table"
          : $"leap seconds: update failed ({ex.Message}); keeping the current table"
      );
      return null;
    }
    finally
    {
      if (tempPath is not null)
        TryDelete(tempPath);
    }
  }

  // ─────────────────────────────────────────────────────── Helpers

  private LeapSecondsLoadStatus? LoadIntoNative(string path, int nn, bool downloaded)
  {
    (int status, uint entries) result;
    try
    {
      result = _loadLeapSecondsFile(path);
    }
    catch (Exception ex)
    {
      LogFailureOnce($"leap seconds: loading {Path.GetFileName(path)} failed ({ex.Message}); keeping the current table");
      return null;
    }

    var status = Enum.IsDefined(typeof(LeapSecondsLoadStatus), result.status)
      ? (LeapSecondsLoadStatus)result.status
      : LeapSecondsLoadStatus.Unreadable;
    var name = Path.GetFileName(path);
    if (status is LeapSecondsLoadStatus.Adopted or LeapSecondsLoadStatus.KeptCurrent)
    {
      _log?.Invoke(
        status == LeapSecondsLoadStatus.Adopted
          ? $"leap seconds: adopted {name} ({result.entries} entries)"
          : $"leap seconds: {name} kept the current table ({result.entries} entries)"
      );
      var args = new LeapSecondsSourceChangedEventArgs(path, nn, status, result.entries, downloaded);
      CurrentSource = args;
      LeapSecondsSourceChanged?.Invoke(this, args);
    }
    else
    {
      _log?.Invoke($"leap seconds: {name} rejected ({status}); keeping the current table");
      try
      {
        var rejected = path + RejectedSuffix;
        if (File.Exists(rejected))
          File.Delete(rejected);
        File.Move(path, rejected);
      }
      catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
      {
        TryDelete(path);
      }
    }
    return status;
  }

  private static async Task<string> ReadCappedStringAsync(HttpResponseMessage response, long cap, CancellationToken ct)
  {
    if (response.Content.Headers.ContentLength is long declared && declared > cap)
      throw new InvalidDataException($"listing is {declared} bytes (cap {cap})");
    using var source = await response.Content.ReadAsStreamAsync().ConfigureAwait(false);
    using var buffer = new MemoryStream();
    await CopyCappedAsync(source, buffer, cap, "listing", ct).ConfigureAwait(false);
    return System.Text.Encoding.UTF8.GetString(buffer.GetBuffer(), 0, (int)buffer.Length);
  }

  private static async Task CopyCappedAsync(Stream source, Stream target, long cap, string what, CancellationToken ct)
  {
    var chunk = new byte[8192];
    long total = 0;
    while (true)
    {
      var read = await source.ReadAsync(chunk, 0, chunk.Length, ct).ConfigureAwait(false);
      if (read == 0)
        return;
      total += read;
      if (total > cap)
        throw new InvalidDataException($"{what} exceeds {cap} bytes");
      await target.WriteAsync(chunk, 0, read, ct).ConfigureAwait(false);
    }
  }

  private void LogFailureOnce(string message)
  {
    if (Interlocked.Exchange(ref _failureLogged, 1) == 0)
      _log?.Invoke(message);
  }

  private static void TryDelete(string path)
  {
    try
    {
      if (File.Exists(path))
        File.Delete(path);
    }
    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
  }
}
