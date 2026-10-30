using System;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using Avalonia.Headless.XUnit;
using Microsoft.Extensions.DependencyInjection;
using Xunit;
using Xunit.Abstractions;

namespace AetherVk.AppTests.Integration;

/// <summary>
/// The UTC ↔ TDB converter over the real core (<c>avkTime_*</c>, hifitime + the leap-second
/// table): the reviewer's FITS mid-exposure 2021-12-27T00:14:47 UTC is 69.184 s later in TDB
/// (ΔAT 37 s + 32.184 s, the periodic term under 2 ms), the round trip lands on the same fields,
/// and the table's last entry is the +37 s of 2017-01-01. Returns early (with a note) when the
/// native runtime is mocked.
/// </summary>
public sealed class TimeScaleNativeTests(ITestOutputHelper output)
{
  private static readonly DateTimeOffset MidUtc = new(2021, 12, 27, 0, 14, 47, TimeSpan.Zero);

  private static INativeRuntimeService? Native(ITestOutputHelper output)
  {
    var services = (App.Host ?? throw new InvalidOperationException("DI host not initialized")).Services;
    var runtime = services.GetRequiredService<INativeRuntimeService>();
    if (runtime is NativeRuntimeService) return runtime;
    output.WriteLine("Skipped: the native runtime is mocked");
    return null;
  }

  [AvaloniaFact]
  public void UtcToTdb_IsTheLeapSecondOffset_AndRoundTrips()
  {
    if (Native(output) is not { } runtime) return;
    var converter = new NativeTimeScaleConverter(runtime);

    var status = converter.UtcToTdbParts(MidUtc, out short c, out ulong ns);
    Assert.Equal(UtcStatus.Ok, status);
    var plain = TimeUtils.ToTdbParts(MidUtc);
    double offset = (TimeUtils.FromTdbParts(c, ns) - TimeUtils.FromTdbParts(plain.centuries, plain.nanoseconds)).TotalSeconds;
    output.WriteLine($"TDB - UTC at {MidUtc:s} = {offset:F6} s");
    Assert.InRange(offset, 69.184 - 0.002, 69.184 + 0.002);
    Assert.InRange(converter.TdbMinusUtcSeconds(c, ns), 69.184 - 0.002, 69.184 + 0.002);

    Assert.Equal(UtcStatus.Ok, converter.TdbPartsToUtc(c, ns, out UtcFields back));
    Assert.Equal(UtcFields.From(MidUtc).ToString(), back.ToString());
    Assert.True(back.Nanos < 1_000u, $"round trip off by {back.Nanos} ns");

    // a leap second: 2016-12-31T23:59:60 is one second after :59 and one before 2017-01-01
    Assert.Equal(UtcStatus.Ok, converter.UtcToTdbParts(new UtcFields(2016, 12, 31, 23, 59, 59), out short c59, out ulong n59));
    Assert.Equal(UtcStatus.Ok, converter.UtcToTdbParts(new UtcFields(2016, 12, 31, 23, 59, 60), out short c60, out ulong n60));
    Assert.Equal(UtcStatus.Ok, converter.UtcToTdbParts(new UtcFields(2017, 1, 1, 0, 0, 0), out short c0, out ulong n0));
    Assert.InRange((TimeUtils.FromTdbParts(c60, n60) - TimeUtils.FromTdbParts(c59, n59)).TotalSeconds, 0.999, 1.001);
    Assert.InRange((TimeUtils.FromTdbParts(c0, n0) - TimeUtils.FromTdbParts(c60, n60)).TotalSeconds, 0.999, 1.001);
    Assert.Equal(UtcStatus.Malformed, converter.UtcToTdbParts(new UtcFields(2021, 12, 27, 0, 14, 60), out _, out _));
  }

  [AvaloniaFact]
  public void LeapSecondsInfo_EndsWithThe2017Entry()
  {
    if (Native(output) is not { } runtime) return;
    var info = new NativeTimeScaleConverter(runtime).LeapSecondsInfo();
    output.WriteLine(info.Summary);
    Assert.Equal(37, info.LastDeltaAt);
    Assert.Equal(new DateTime(2017, 1, 1), info.LastEntry.Date);
    Assert.True(info.EntryCount >= 28, $"{info.EntryCount} entries");
    Assert.EndsWith("last +37 s on 2017-01-01", info.Summary, StringComparison.Ordinal);

    // beyond the validity horizon: converted, flagged
    var svc = new TimeScaleService(new NativeTimeScaleConverter(runtime));
    Assert.True(svc.TryParse("2040-01-01T00:00:00", TimeScale.Utc, out _, out _, out var status));
    Assert.Equal(UtcStatus.BeyondLskValidity, status);
  }
}
