using System.Text.Json;
using System.Text.Json.Serialization;

namespace AetherVk.UI.E2E;

/// <summary>
/// The reference machine of the playback perf limits (perf_reference.json, next to the test
/// assembly): its GPU, the particle quality of the calibration, and the median dust frame time
/// it measured there, paused. <c>null</c> <see cref="CalibTotalMs"/>: not recorded yet.
/// </summary>
public sealed record PerfReference(
  [property: JsonPropertyName("gpu")] string Gpu,
  [property: JsonPropertyName("quality")] string Quality,
  [property: JsonPropertyName("calib_total_ms")] double? CalibTotalMs,
  [property: JsonPropertyName("score_min")] double ScoreMin,
  [property: JsonPropertyName("score_max")] double ScoreMax)
{
  public const string FileName = "perf_reference.json";

  /// <summary>With this env var at 1 the perf test records a suggested reference and asserts nothing.</summary>
  public const string RecordEnvVar = "AETHERVK_E2E_PERF_RECORD";

  public static bool Recording => Environment.GetEnvironmentVariable(RecordEnvVar) == "1";

  private static readonly JsonSerializerOptions Options = new() { WriteIndented = true };

  public static PerfReference Load()
  {
    var path = Path.Combine(AppContext.BaseDirectory, FileName);
    return JsonSerializer.Deserialize<PerfReference>(File.ReadAllText(path))
      ?? throw new InvalidDataException($"{path} is empty");
  }

  public string ToJson() => JsonSerializer.Serialize(this, Options);
}

/// <summary>
/// How fast this machine draws the dust against the reference (a GTX 1070): reference frame
/// time ÷ measured, within the reference's clamp. 1 on the reference, about 0.3 on an iGPU; the
/// perf limits of <c>PlaybackPerfTests</c> scale by it, so the same test holds on both.
/// </summary>
public sealed record Capability(string Gpu, double CalibTotalMs, double Score, bool Referenced)
{
  public static Capability Measure(PerfReference reference, string gpu, double calibTotalMs)
  {
    if (reference.CalibTotalMs is not { } refMs || !(calibTotalMs > 0))
      return new Capability(gpu, calibTotalMs, 1.0, Referenced: false);
    return new Capability(gpu, calibTotalMs, Math.Clamp(refMs / calibTotalMs, reference.ScoreMin, reference.ScoreMax), true);
  }

  public override string ToString() =>
    $"GPU \"{Gpu}\", calibration {CalibTotalMs:0.00} ms, capability score {Score:0.00}" +
    (Referenced ? string.Empty : " (no reference recorded: score 1)");
}
