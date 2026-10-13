using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using Avalonia.Headless.XUnit;
using Microsoft.Extensions.DependencyInjection;
using Xunit;
using Xunit.Abstractions;

namespace AetherVk.AppTests.Integration;

/// <summary>
/// The savefile end to end on the real engine (docs/savefile.md, "Verification"): the 67P SPK
/// fixture in <c>test_assets/spk/</c> (no network) is committed with a jet, the clock moved, the
/// session saved; File › New; the save reopened restores the cached scene (same compatibility
/// JSON, same epoch, the jet with its seed); a radius change saves a second state; 11 h later the
/// stale one expires. Returns early (with a note) when the native runtime is mocked.
/// </summary>
public sealed class SaveFileNativeE2ETests(ITestOutputHelper output)
{
  private sealed class Clock(DateTimeOffset now) : ISystemClock
  {
    public DateTimeOffset UtcNow { get; } = now;
  }

  private static string? Fixture()
  {
    for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
    {
      var f = Path.Combine(dir.FullName, "test_assets", "spk", "spk_67P_90000703.bsp");
      if (File.Exists(f)) return f;
    }
    return null;
  }

  private static EpochParts Parts(string iso)
  {
    Assert.True(TimeUtils.TryParseIso8601(iso, out var t));
    var (c, ns) = TimeUtils.ToTaiParts(t);
    return new EpochParts(c, ns);
  }

  private static double SecondsBetween(EpochParts a, short c, ulong ns) =>
    Math.Abs((TimeUtils.FromTaiParts(a.Centuries, a.Nanoseconds) - TimeUtils.FromTaiParts(c, ns)).TotalSeconds);

  [AvaloniaFact]
  public async Task Commit_Save_New_Open_RestoresTheSameScene_ThenExpires()
  {
    var services = (App.Host ?? throw new InvalidOperationException("DI host not initialized")).Services;
    var runtime = services.GetRequiredService<INativeRuntimeService>();
    if (runtime is not NativeRuntimeService || services.GetService<SaveFileService>() is not { } saves)
    {
      output.WriteLine("Skipped: the native runtime is mocked");
      return;
    }
    var fixture = Fixture() ?? throw new FileNotFoundException("test_assets/spk/spk_67P_90000703.bsp");
    var host = services.GetRequiredService<ISaveFileHost>();
    var cometConfig = services.GetRequiredService<CometConfigService>();
    var model = services.GetRequiredService<ITabStateService<ModelSession>>();
    var modelSession = model.GetSession(model.ActiveSessionIds[0]);

    var root = Path.Combine(Path.GetTempPath(), "avk_savefile_e2e_" + Guid.NewGuid().ToString("N"));
    var downloads = Directory.CreateDirectory(Path.Combine(root, "downloads")).FullName;
    var spkPath = Path.Combine(downloads, "spk_67P_90000703.bsp");
    File.Copy(fixture, spkPath);
    var saveDir = Path.Combine(root, "save");
    try
    {
      await saves.NewAsync();

      // ── a session: one jet, the comet committed from the "downloaded" kernel, the clock moved ──
      host.ApplyModel(new ModelDto(), [new JetDto
      {
        LatitudeDeg = 10, LongitudeDeg = 20, ApertureDeg = 30, VelocityMeanMs = 2, VelocityStdMs = 0.5f, Seed = 4242,
      }]);
      host.ApplyTimeline(new TimelineDto { ProposedStart = "2025-10-02T00:00:00Z", ProposedEnd = "2025-11-10T00:00:00Z" },
        committed: false);
      var start = Parts("2025-10-02T00:00:00Z");
      var end = Parts("2025-11-10T00:00:00Z");
      var spk = new SpkDto
      {
        NaifId = 1000012, Designation = "67P", FullName = "67P/Churyumov-Gerasimenko", RecordId = "90000703",
        Sbdb = new SbdbDto { SpkId = 1000012, Designation = "67P", FullName = "67P/Churyumov-Gerasimenko" },
        ReferenceOrbitMode = "osculating_at_start", NucleusRadiusKm = 2.0f,
      };
      Assert.True(await host.CommitCometAsync(spkPath, spk, SaveIndexMapping.ToRange(start, end)), "commit");
      var jet = Assert.Single(modelSession.Jets);
      Assert.NotEqual(0ul, jet.NativePsId);
      var mid = Parts("2025-10-20T06:00:00Z");
      Assert.True(runtime.SeekEpochSync(mid.Centuries, mid.Nanoseconds));
      await host.SettleAsync();
      var compat = runtime.GetCompatibilityJson();
      Assert.NotNull(compat);

      // ── save ──
      Assert.True(await saves.SaveAsync(saveDir), "save");
      var (index, err) = SaveIndexIo.Read(saveDir);
      Assert.Equal(SaveIndexError.None, err);
      var state = Assert.Single(index!.States);
      Assert.Equal(SaveIndexIo.CompatHash(compat!), state.CompatHash);
      Assert.Equal(runtime.SceneDumpVersion, state.DumpVersion);
      Assert.True(new FileInfo(Path.Combine(saveDir, state.File)).Length > 0);
      Assert.Equal(File.ReadAllBytes(fixture), File.ReadAllBytes(Path.Combine(saveDir, Assert.Single(index.Spk).File)));
      Assert.Equal(4242u, Assert.Single(index.Jets).Seed);
      Assert.NotNull(index.Timeline.CurrentEpoch);
      Assert.True(SecondsBetween(index.Timeline.CurrentEpoch!, mid.Centuries, mid.Nanoseconds) < 1);

      // ── New: nothing committed ──
      await saves.NewAsync();
      Assert.False(cometConfig.IsAlmanacCommittedValue);
      Assert.Empty(modelSession.Jets);

      // ── open: the cached state is restored ──
      var opened = await saves.OpenAsync(saveDir);
      Assert.True(opened.Success);
      Assert.True(opened.StateRestored,
        $"the cached scene state was not restored\nsaved: {compat}\nlive:  {runtime.GetCompatibilityJson()}");
      Assert.True(cometConfig.IsAlmanacCommittedValue);
      Assert.Equal(compat, runtime.GetCompatibilityJson());
      Assert.True(runtime.GetSimulationClock(out var c, out var ns));
      Assert.True(SecondsBetween(mid, c, ns) < 1, "clock not at the saved epoch");
      jet = Assert.Single(modelSession.Jets);
      Assert.Equal(4242u, jet.Seed);
      Assert.NotEqual(0ul, jet.NativePsId);
      Assert.StartsWith(Path.GetFullPath(saveDir), cometConfig.CommittedInputs!.SpkFilePath);
      Assert.False(saves.IsDirty);

      // ── a second configuration (radius and a rotation, as the Comet tab sets them): a second state ──
      var comets = services.GetRequiredService<ITabStateService<CometSession>>();
      var cometSession = comets.GetSession(comets.ActiveSessionIds[0]);
      var rotation = new RotationDto { PoleRaDeg = 69.3, PoleDecDeg = 64.1, RateDegDay = 649.1, BodyFixedOrientation = true };
      SaveIndexMapping.Apply(rotation, cometSession);
      cometConfig.SetRotationalModel(SaveIndexMapping.ToNative(rotation));
      cometSession.NucleusRadiusKm = 3.0f;
      cometConfig.SetNucleusRadiusKm(3.0f);
      await host.SettleAsync();
      Assert.NotEqual(compat, runtime.GetCompatibilityJson());
      Assert.True(await saves.SaveAsync(saveDir));
      (index, _) = SaveIndexIo.Read(saveDir);
      Assert.Equal(2, index!.States.Count);
      Assert.Equal(1, index.States.Count(s => s.Latest));
      Assert.NotEqual(state.CompatHash, index.States.Single(s => s.Latest).CompatHash);
      Assert.Equal(69.3, Assert.Single(index.Spk).Rotation!.PoleRaDeg);
      var compat2 = runtime.GetCompatibilityJson();

      // ── 11 h later: the unused, non-latest state expires with its file ──
      var later = new SaveFileService(host, new Clock(DateTimeOffset.UtcNow + TimeSpan.FromHours(11)));
      var reopened = await later.OpenAsync(saveDir);
      Assert.True(reopened.Success);
      Assert.True(reopened.StateRestored,
        $"the rotated configuration's state was not restored\nsaved: {compat2}\nlive:  {runtime.GetCompatibilityJson()}");
      Assert.Equal(compat2, runtime.GetCompatibilityJson());
      Assert.Equal(1, reopened.ExpiredStates);
      (index, _) = SaveIndexIo.Read(saveDir);
      Assert.NotEqual(state.CompatHash, Assert.Single(index!.States).CompatHash);
      Assert.False(File.Exists(Path.Combine(saveDir, state.File)));
    }
    finally
    {
      await saves.NewAsync();
      try { Directory.Delete(root, recursive: true); } catch (IOException) { }
    }
  }
}
