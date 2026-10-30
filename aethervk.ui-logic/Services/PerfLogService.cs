using System;
using System.Globalization;
using System.Threading;

namespace AetherVk.Logic.Services;

/// <summary>
/// With <c>AETHERVK_PERF_LOG=1</c>, one <c>[perf] {json}</c> line per second on stdout: the dust
/// GPU timings of the main view (its frame counter gives the dust frame rate), the simulation
/// clock in seconds (the engine's epoch duration: compare differences only) and whether the dust
/// is drawn in preview. The playback e2e reads these lines live (<c>PlaybackPerfTests</c>); off
/// by default, nothing runs.
/// </summary>
public sealed class PerfLogService : IDisposable
{
  public const string EnvVar = "AETHERVK_PERF_LOG";

  private const double SecondsPerCentury = 36525.0 * 86400.0;

  private readonly INativeRuntimeService _runtime;
  private readonly Action<string> _write;
  private Timer? _timer;
  private readonly System.Diagnostics.Stopwatch _clock = System.Diagnostics.Stopwatch.StartNew();

  public PerfLogService(INativeRuntimeService runtime, Action<string>? write = null)
  {
    _runtime = runtime;
    _write = write ?? Console.WriteLine;
  }

  public static bool Enabled => Environment.GetEnvironmentVariable(EnvVar) == "1";

  public void Start(TimeSpan? period = null)
  {
    var p = period ?? TimeSpan.FromSeconds(1);
    _timer ??= new Timer(_ => Tick(), null, p, p);
  }

  /// <summary>Writes one line (the timer's thread: the queries take the scene lock).</summary>
  public void Tick()
  {
    try
    {
      _write(Line());
    }
    catch (Exception e)
    {
      _write($"[perf] error {e.GetType().Name}: {e.Message}");
    }
  }

  internal string Line()
  {
    var ci = CultureInfo.InvariantCulture;
    string Num(double v) => !double.IsNaN(v) && !double.IsInfinity(v) ? v.ToString("0.###", ci) : "null";
    var epoch = _runtime.GetSimulationClock(out var c, out var ns) ? Num(c * SecondsPerCentury + ns * 1e-9) : "null";
    var preview = _runtime.DustInPreview() switch { true => "true", false => "false", null => "null" };
    // the viewport badge's phase: playing / hold (with its progress) / rendering / settled
    var state = _runtime.DustRenderState();
    var phase = state is { } st
      ? $"\"phase\":\"{st.Phase.ToString().ToLowerInvariant()}\",\"hold\":{Num(st.HoldProgress)}"
      : "\"phase\":null";
    var pe = _runtime.PresentationEngineId;
    var t = pe is { } p ? _runtime.DustGpuTimings(p) : null;
    // the last frame that drew (not a reused image): a settled view's cost, still there long after
    var d = pe is { } q ? _runtime.DustGpuTimingsDrawn(q) : null;
    var drawn = d is null
      ? ""
      : $",\"drawn_frame\":{d.Frame},\"drawn_total_ms\":{Num(d.TotalMs)},\"drawn_splat_ms\":{Num(d.SplatMs)},\"drawn_stride\":{d.Stride}";
    var timings = t is null
      ? "\"dust_frame\":null"
      : $"\"dust_frame\":{t.Frame},\"total_ms\":{Num(t.TotalMs)},\"total_ema_ms\":{Num(t.TotalEmaMs)}," +
        $"\"propagate_ms\":{Num(t.PropagateMs)},\"splat_ms\":{Num(t.SplatMs)},\"measure_ms\":{Num(t.MeasureMs)}," +
        $"\"composite_ms\":{Num(t.CompositeMs)},\"w\":{t.Width},\"h\":{t.Height},\"stride\":{t.Stride}," +
        $"\"specks_ms\":{Num(t.SpecksMs)},\"splat_only_ms\":{Num(t.SplatOnlyMs)},\"queue_ms\":{Num(t.QueueMs)}";
    return $"[perf] {{\"t_ms\":{_clock.ElapsedMilliseconds},\"epoch_s\":{epoch},\"preview\":{preview},{phase},{timings}{drawn}}}";
  }

  public void Dispose()
  {
    _timer?.Dispose();
    _timer = null;
  }
}
