using System.Text.Json;
using System.Text.RegularExpressions;

namespace AetherVk.UI.E2E;

/// <summary>
/// One <c>[perf] {json}</c> line of the app (<c>PerfLogService</c>, <c>AETHERVK_PERF_LOG=1</c>):
/// its own wall clock in ms, the simulation clock in seconds (differences only), whether the dust
/// was drawn in preview, and the main view's dust GPU timings with their frame counter.
/// </summary>
public sealed record PerfSample(
  DateTime At, double TMs, double? EpochS, bool? Preview, ulong? DustFrame,
  double? TotalMs, double? TotalEmaMs, double? PropagateMs, double? SplatMs, double? MeasureMs, double? CompositeMs,
  int? Width, int? Height, int? Stride,
  ulong? DrawnFrame = null, double? DrawnTotalMs = null, double? DrawnSplatMs = null, int? DrawnStride = null,
  double? SpecksMs = null, double? SplatOnlyMs = null, double? QueueMs = null,
  string? Phase = null, double? Hold = null)
{
  public static string CsvHeader =>
    "t_ms,epoch_s,preview,dust_frame,total_ms,total_ema_ms,propagate_ms,splat_ms,measure_ms,composite_ms,w,h,stride,specks_ms,splat_only_ms,queue_ms";

  public string Csv() => string.Join(",", new object?[]
    { TMs, EpochS, Preview, DustFrame, TotalMs, TotalEmaMs, PropagateMs, SplatMs, MeasureMs, CompositeMs, Width, Height, Stride, SpecksMs, SplatOnlyMs, QueueMs }
    .Select(v => v switch { null => "", double d => d.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture), _ => v.ToString() }));
}

/// <summary>Reads the app's <c>[perf]</c> lines and the GPU it runs on from <see cref="AppSession.Lines"/>.</summary>
public static partial class PerfLog
{
  private const string Prefix = "[perf] {";

  /// <summary>The perf lines written since <paramref name="since"/>, in order.</summary>
  public static List<PerfSample> Since(AppSession s, DateTime since) =>
    s.Lines.Where(l => l.At >= since).Select(l => Parse(l.At, l.Line)).OfType<PerfSample>().ToList();

  /// <summary>Waits until <paramref name="count"/> perf lines were written since <paramref name="since"/>.</summary>
  public static List<PerfSample> Collect(AppSession s, DateTime since, int count, TimeSpan? timeout = null)
  {
    var limit = DateTime.UtcNow + (timeout ?? TimeSpan.FromSeconds(count * 2 + 10));
    while (true)
    {
      var got = Since(s, since);
      if (got.Count >= count)
        return got.Take(count).ToList();
      s.EnsureAlive("waiting for the perf lines");
      if (DateTime.UtcNow > limit)
        throw new TimeoutException($"{got.Count} of {count} [perf] lines (was the app launched with perfLog: true?)");
      Thread.Sleep(200);
    }
  }

  /// <summary>The lines containing <paramref name="text"/> written since <paramref name="since"/>.</summary>
  public static int Count(AppSession s, DateTime since, string text) =>
    s.Lines.Count(l => l.At >= since && l.Line.Contains(text, StringComparison.Ordinal));

  /// <summary>The name of the GPU the app picked (the core's "[Device] eligible GPU 0" line), or "unknown".</summary>
  public static string GpuName(AppSession s)
  {
    foreach (var (_, line) in s.Lines)
    {
      var m = GpuRegex().Match(line);
      if (m.Success)
        return m.Groups["name"].Value;
    }
    return "unknown";
  }

  [GeneratedRegex("\\[Device\\] eligible GPU 0: \"(?<name>[^\"]*)\"")]
  private static partial Regex GpuRegex();

  public static PerfSample? Parse(DateTime at, string line)
  {
    int i = line.IndexOf(Prefix, StringComparison.Ordinal);
    if (i < 0)
      return null;
    try
    {
      using var doc = JsonDocument.Parse(line[(i + Prefix.Length - 1)..]);
      var r = doc.RootElement;
      double? D(string name) => r.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : null;
      bool? B(string name) => r.TryGetProperty(name, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False ? v.GetBoolean() : null;
      ulong? U(string name) => r.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetUInt64() : null;
      string? S(string name) => r.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
      return new PerfSample(at, D("t_ms") ?? double.NaN, D("epoch_s"), B("preview"), U("dust_frame"),
        D("total_ms"), D("total_ema_ms"), D("propagate_ms"), D("splat_ms"), D("measure_ms"), D("composite_ms"),
        (int?)D("w"), (int?)D("h"), (int?)D("stride"),
        U("drawn_frame"), D("drawn_total_ms"), D("drawn_splat_ms"), (int?)D("drawn_stride"),
        D("specks_ms"), D("splat_only_ms"), D("queue_ms"), S("phase"), D("hold"));
    }
    catch (JsonException)
    {
      return null;
    }
  }
}
