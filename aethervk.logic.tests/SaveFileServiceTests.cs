using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using AetherVk.Logic.Tests.Mocks;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// <see cref="SaveFileService"/> against a fake application (docs/savefile.md): what Save writes,
/// that Open rebuilds the same state, the cached-state rules (restore vs seek), the expiry matrix
/// with an injected clock, asset export / cleanup / fallback and key remapping, dirty tracking.
/// </summary>
public sealed class SaveFileServiceTests : IDisposable
{
  private readonly string _root = Path.Combine(Path.GetTempPath(), "avk_savefile_" + Guid.NewGuid().ToString("N"));

  public SaveFileServiceTests() => Directory.CreateDirectory(_root);

  public void Dispose()
  {
    try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
  }

  private sealed class Clock : ISystemClock
  {
    public DateTimeOffset UtcNow { get; set; } = new(2026, 10, 13, 12, 0, 0, TimeSpan.Zero);
  }

  private static SpkDto Spk67P(float radius = 2f) => new()
  {
    NaifId = 1000012, Designation = "67P", FullName = "67P/Churyumov-Gerasimenko", RecordId = "90000703",
    Sbdb = new SbdbDto { SpkId = 1000012, E = 0.64, Q = 1.24 }, NucleusRadiusKm = radius,
    Rotation = new RotationDto { PoleRaDeg = 69.3, PoleDecDeg = 64.1, RateDegDay = 649.1 },
  };

  /// <summary>A host with a committed comet, a jet, a GLB (with an embedded texture), a png, a billboard.</summary>
  private FakeSaveFileHost Populated()
  {
    var h = new FakeSaveFileHost(_root);
    Directory.CreateDirectory(Path.GetDirectoryName(h.SpkSource)!);
    File.WriteAllText(h.SpkSource, "DAF/SPK 67P");
    h.Comet = Spk67P();
    h.Timeline.CommittedStart = new EpochParts(0, 1000);
    h.Timeline.CommittedEnd = new EpochParts(0, 9000);
    h.Timeline.CurrentEpoch = new EpochParts(0, 4000);
    h.Model = new ModelDto { DiametreUm = 250, ManualNucleusRadiusKm = 3.5f };
    h.Jets = [new JetDto { LatitudeDeg = 10, LongitudeDeg = 20, ApertureDeg = 30, Seed = 77 }];
    h.Viewports = [new ViewportDto { Mode = "comet_orbiting", OrbitOffsetAu = [0.001, 0, 0] }];
    var mesh = h.AddAsset(AssetKind.Mesh, "/home/u/rock.glb", "rock", "MESH|rock-albedo-texels");
    var png = h.AddAsset(AssetKind.Texture, "/home/u/sky.png", "sky", "sky-texels");
    var embedded = h.Assets.Single(a => a.Key == "/home/u/rock.glb#image0");
    h.Appearance = new CometAppearanceDto { Mesh = mesh.Id, Albedo = embedded.Id, Yaw = 15, Mode = CometDisplayMode.Custom };
    h.Billboards = [new BillboardDto { Id = Guid.NewGuid(), TextureKey = png.Key, CenterX = 100, CenterY = 200, WidthPx = 64, HeightPx = 32 }];
    return h;
  }

  private string SaveDir => Path.Combine(_root, "save");

  [Fact]
  public async Task Save_WritesSpkAssetsStateAndIndex_Atomically()
  {
    var host = Populated();
    var svc = new SaveFileService(host, new Clock(), appVersion: "test");
    Assert.True(await svc.SaveAsync(SaveDir));

    var (index, err) = SaveIndexIo.Read(SaveDir);
    Assert.Equal(SaveIndexError.None, err);
    Assert.False(File.Exists(SaveIndexIo.IndexPath(SaveDir) + ".tmp"));
    var spk = Assert.Single(index!.Spk);
    Assert.True(spk.Committed);
    Assert.Equal("spk/spk_67P_90000703.bsp", spk.File);
    Assert.Equal("DAF/SPK 67P", File.ReadAllText(Path.Combine(SaveDir, spk.File)));
    Assert.Equal(3, index.Assets.Count);
    foreach (var a in index.Assets)
      Assert.True(File.Exists(Path.Combine(SaveDir, a.StoredFile)), a.StoredFile);
    Assert.Equal("/home/u/rock.glb", index.Assets.Single(a => a.Kind == "mesh").OriginalSource);
    var state = Assert.Single(index.States);
    Assert.True(state.Latest);
    Assert.Equal(8u, state.DumpVersion);
    Assert.Equal(SaveIndexIo.CompatHash(host.GetCompatibilityJson()!), state.CompatHash);
    Assert.Equal($"states/{state.CompatHash}/scene.bin", state.File);
    Assert.True(File.Exists(Path.Combine(SaveDir, state.File)));
    Assert.Equal("custom", index.CometAppearance!.Mode);
    Assert.Equal("/home/u/rock.glb#image0", index.CometAppearance.AlbedoKey);
    Assert.Equal(250, index.Model.DiametreUm);
    Assert.Equal(77u, Assert.Single(index.Jets).Seed);
    Assert.Equal("test", index.AppVersion);
    Assert.Equal(SaveDir, svc.CurrentSaveDir);
    Assert.False(svc.IsDirty);
  }

  [Fact]
  public async Task Open_RebuildsTheSameState_AndRestoresTheCachedScene()
  {
    var saved = Populated();
    var original = saved.Capture();
    await new SaveFileService(saved, new Clock()).SaveAsync(SaveDir);

    var host = new FakeSaveFileHost(_root);
    var svc = new SaveFileService(host, new Clock());
    var r = await svc.OpenAsync(SaveIndexIo.IndexPath(SaveDir));
    Assert.True(r.Success);
    Assert.True(r.StateRestored);
    Assert.False(r.Sought);
    Assert.Empty(r.MissingFiles!);
    // the timeline shows the restored epoch, and the dust feedback restarts on the settled view
    Assert.Equal(1, host.ClockSyncs);
    Assert.Equal(1, host.DustFeedbackRestarts);

    // meshes first; the embedded texture came with the GLB, so only the png is imported besides it
    var imports = host.Calls.Where(c => c.StartsWith("import", StringComparison.Ordinal)).ToList();
    Assert.Equal(2, imports.Count);
    Assert.EndsWith(".glb", imports[0]);
    Assert.Contains("commit spk/spk_67P_90000703.bsp", host.Calls.Select(c => c.Replace(Path.DirectorySeparatorChar, '/').Replace("save/", "")));
    Assert.Equal(4000ul, host.Timeline.CurrentEpoch!.Nanoseconds);

    var now = host.Capture();
    Assert.Equal(SaveIndexIo.ToJson(new SaveIndex { Model = original.Model }), SaveIndexIo.ToJson(new SaveIndex { Model = now.Model }));
    Assert.Equal(77u, Assert.Single(now.Jets).Seed);
    Assert.Equal("comet_orbiting", Assert.Single(now.Viewports).Mode);
    // the appearance names the re-imported assets
    Assert.Equal(CometDisplayMode.Custom, host.Appearance.Mode);
    Assert.Equal(host.Assets.Single(a => a.Kind == AssetKind.Mesh).Id, host.Appearance.Mesh);
    Assert.Equal(host.Assets.Single(a => a.Key.EndsWith("#image0", StringComparison.Ordinal)).Id, host.Appearance.Albedo);
    Assert.Equal(15, host.Appearance.Yaw);
    // the billboard follows its texture's new key
    var png = host.Assets.Single(a => a.Label == "sky");
    Assert.Equal(png.Key, Assert.Single(host.Billboards).TextureKey);
    Assert.Equal(original.Billboards[0].Id, host.Billboards[0].Id);
    Assert.False(svc.IsDirty);

    // saving the reopened session keeps the original sources and reuses the stored files
    await svc.SaveAsync(SaveDir);
    var (index, _) = SaveIndexIo.Read(SaveDir);
    Assert.Equal(3, index!.Assets.Count);
    Assert.Equal("/home/u/sky.png", index.Assets.Single(a => a.Label == "sky").OriginalSource);
    Assert.Single(index.States);
  }

  [Theory]
  [InlineData("version")]
  [InlineData("missing")]
  [InlineData("hash")]
  public async Task Open_Seeks_WhenTheCachedStateDoesNotMatch(string why)
  {
    var saved = Populated();
    await new SaveFileService(saved, new Clock()).SaveAsync(SaveDir);
    var (index, _) = SaveIndexIo.Read(SaveDir);
    var host = new FakeSaveFileHost(_root);
    switch (why)
    {
      case "version":
        host.DumpVersion = 9;
        break;
      case "missing":
        File.Delete(Path.Combine(SaveDir, index!.States[0].File));
        break;
      case "hash":
        index!.States[0].CompatHash = "0000000000000000";
        SaveIndexIo.Write(SaveDir, index);
        break;
    }
    var r = await new SaveFileService(host, new Clock()).OpenAsync(SaveDir);
    Assert.True(r.Success);
    Assert.False(r.StateRestored);
    Assert.True(r.Sought);
    Assert.DoesNotContain(host.Calls, c => c.StartsWith("restore", StringComparison.Ordinal));
    Assert.Contains("seek 4000", host.Calls);
  }

  /// <summary>
  /// Expiry at open: only entries that are both unused for more than 10 h and not current
  /// (latest state / committed SPK) go, with their files.
  /// </summary>
  [Fact]
  public async Task Open_ExpiresStaleEntries_WithTheirFiles()
  {
    var clock = new Clock();
    var host = Populated();
    var svc = new SaveFileService(host, clock);
    await svc.SaveAsync(SaveDir);                       // state A at t0
    clock.UtcNow += TimeSpan.FromHours(1);
    host.Jets.Add(new JetDto { LatitudeDeg = 50, Seed = 1 });
    await svc.SaveAsync(SaveDir);                       // state B at t0+1h (latest)
    clock.UtcNow += TimeSpan.FromHours(1);
    host.Comet = Spk67P(radius: 3);
    await svc.SaveAsync(SaveDir);                       // state C at t0+2h (latest)

    // a second, uncommitted SPK and an extra recent state
    var (index, _) = SaveIndexIo.Read(SaveDir);
    Assert.Equal(3, index!.States.Count);
    File.WriteAllText(Path.Combine(SaveDir, "spk", "old.bsp"), "old");
    index.Spk.Add(new SpkDto { File = "spk/old.bsp", LastUsedAt = clock.UtcNow - TimeSpan.FromHours(20) });
    index.Spk.Single(s => s.Committed).LastUsedAt = clock.UtcNow - TimeSpan.FromHours(30);
    SaveIndexIo.Write(SaveDir, index);
    var stateA = index.States.OrderBy(s => s.CreatedAt).First();
    var stateB = index.States.OrderBy(s => s.CreatedAt).ElementAt(1);
    var stateC = index.States.Single(s => s.Latest);

    clock.UtcNow = stateA.LastUsedAt + TimeSpan.FromHours(10.5); // A: 10.5 h, B: 9.5 h, C: 8.5 h
    var r = await new SaveFileService(new FakeSaveFileHost(_root), clock).OpenAsync(SaveDir);
    Assert.Equal(1, r.ExpiredStates);
    Assert.Equal(1, r.ExpiredSpk);
    (index, _) = SaveIndexIo.Read(SaveDir);
    Assert.Equal([stateB.CompatHash, stateC.CompatHash], index!.States.Select(s => s.CompatHash).OrderBy(h => h == stateC.CompatHash));
    Assert.False(Directory.Exists(Path.Combine(SaveDir, "states", stateA.CompatHash)));
    Assert.True(File.Exists(Path.Combine(SaveDir, stateB.File)));
    Assert.False(File.Exists(Path.Combine(SaveDir, "spk", "old.bsp")));
    // the committed SPK stays however old it is
    Assert.True(File.Exists(Path.Combine(SaveDir, "spk", "spk_67P_90000703.bsp")));

    // the latest state is never expired
    clock.UtcNow += TimeSpan.FromDays(30);
    r = await new SaveFileService(new FakeSaveFileHost(_root), clock).OpenAsync(SaveDir);
    Assert.Equal(1, r.ExpiredStates); // B
    (index, _) = SaveIndexIo.Read(SaveDir);
    Assert.Equal(stateC.CompatHash, Assert.Single(index!.States).CompatHash);
  }

  [Fact]
  public async Task Save_DeletesTheFilesOfUnloadedAssets()
  {
    var host = Populated();
    var svc = new SaveFileService(host, new Clock());
    await svc.SaveAsync(SaveDir);
    var (before, _) = SaveIndexIo.Read(SaveDir);
    var sky = before!.Assets.Single(a => a.Label == "sky");
    host.Assets.RemoveAll(a => a.Label == "sky");
    host.Billboards.Clear();
    await svc.SaveAsync(SaveDir);
    var (after, _) = SaveIndexIo.Read(SaveDir);
    Assert.DoesNotContain(after!.Assets, a => a.Label == "sky");
    Assert.False(File.Exists(Path.Combine(SaveDir, sky.StoredFile)));
    Assert.Equal(2, Directory.GetFiles(Path.Combine(SaveDir, "assets"), "*", SearchOption.AllDirectories).Length);
  }

  [Fact]
  public async Task Open_MissingMesh_FallsBackToTheDefaultNucleus()
  {
    await new SaveFileService(Populated(), new Clock()).SaveAsync(SaveDir);
    var (index, _) = SaveIndexIo.Read(SaveDir);
    File.Delete(Path.Combine(SaveDir, index!.Assets.Single(a => a.Kind == "mesh").StoredFile));
    var host = new FakeSaveFileHost(_root);
    var r = await new SaveFileService(host, new Clock()).OpenAsync(SaveDir);
    Assert.True(r.Success);
    Assert.Contains(r.MissingFiles!, f => f.EndsWith(".glb", StringComparison.Ordinal));
    Assert.Equal(CometDisplayMode.Default, host.Appearance.Mode);
    Assert.Contains(host.Notes, n => n.Contains("default nucleus", StringComparison.Ordinal));
    // the embedded albedo was saved on its own: it still loads, and the billboard texture too
    Assert.Equal(2, host.Assets.Count);
  }

  [Fact]
  public async Task Open_RefusesANewerFormat_AndAMissingIndex()
  {
    var host = new FakeSaveFileHost(_root);
    var svc = new SaveFileService(host, new Clock());
    var r = await svc.OpenAsync(SaveDir);
    Assert.False(r.Success);
    Assert.Equal(SaveIndexError.Missing, r.Error);

    Directory.CreateDirectory(SaveDir);
    File.WriteAllText(SaveIndexIo.IndexPath(SaveDir), "{\"format_version\": 99}");
    r = await svc.OpenAsync(SaveDir);
    Assert.Equal(SaveIndexError.NewerFormat, r.Error);
    Assert.DoesNotContain("reset", host.Calls);
    Assert.Null(svc.CurrentSaveDir);
  }

  [Fact]
  public async Task Save_KeepsThePreviousIndex_WhenTheDumpFails_AndMarksNoState()
  {
    var host = Populated();
    var svc = new SaveFileService(host, new Clock());
    host.DumpFails = true;
    Assert.True(await svc.SaveAsync(SaveDir));
    var (index, _) = SaveIndexIo.Read(SaveDir);
    Assert.Empty(index!.States);
    Assert.Single(host.Notes);
  }

  [Fact]
  public async Task Dirty_FollowsChanges_ButNotTheClockOrCameraPose()
  {
    var host = Populated();
    var svc = new SaveFileService(host, new Clock());
    await svc.SaveAsync(SaveDir);
    Assert.False(svc.IsDirty);
    host.Timeline.CurrentEpoch = new EpochParts(0, 5000);
    host.Viewports[0].Camera = new CameraSnapshotDto { Transform = [1, 2, 3, 0, 0, 0, 1] };
    Assert.False(svc.IsDirty);
    host.Model.DiametreUm = 300;
    Assert.True(svc.IsDirty);
    await svc.SaveAsync(SaveDir);
    Assert.False(svc.IsDirty);
    host.Billboards[0].Alpha = 0.9;
    Assert.True(svc.IsDirty);
    await svc.NewAsync();
    Assert.False(svc.IsDirty);
    Assert.Null(svc.CurrentSaveDir);
  }

  /// The startup baseline can be taken before the first viewport exists: a fresh viewport appearing
  /// later is no change (the reopen then asked to save an untouched "Untitled").
  [Fact]
  public void Dirty_IgnoresAFreshViewport_AddedAfterTheBaseline()
  {
    var host = new FakeSaveFileHost(_root) { Viewports = [] };
    var svc = new SaveFileService(host, new Clock());
    Assert.False(svc.IsDirty);
    var northUp = SaveIndexMapping.Snake(ObserverUp.CelestialNorth);
    host.Viewports = [new ViewportDto { Observer = new ObserverDto { Up = northUp } }];
    Assert.False(svc.IsDirty);
    host.Viewports[0].Observer!.LatDeg = 45;
    Assert.True(svc.IsDirty);
  }

  /// An otherwise untouched viewport switched to the ecliptic up is a change, kept in the save
  /// (dropped as fresh, it reloaded north-up).
  [Fact]
  public void Dirty_SeesAnUntouchedViewportSwitchedToEclipticUp()
  {
    var host = new FakeSaveFileHost(_root) { Viewports = [] };
    var svc = new SaveFileService(host, new Clock());
    host.Viewports = [new ViewportDto { Observer = new ObserverDto { Up = SaveIndexMapping.Snake(ObserverUp.CelestialNorth) } }];
    Assert.False(svc.IsDirty);
    host.Viewports[0].Observer!.Up = SaveIndexMapping.Snake(ObserverUp.Ecliptic);
    Assert.True(svc.IsDirty);
  }

  /// <summary>Runtime reuse: going back to a saved configuration restores its state; not while playing.</summary>
  [Fact]
  public async Task ReuseState_RestoresAMatchingSavedState()
  {
    var host = Populated();
    var svc = new SaveFileService(host, new Clock());
    await svc.SaveAsync(SaveDir);                     // state for radius 2
    host.Comet = Spk67P(radius: 3);
    host.Timeline.CurrentEpoch = new EpochParts(0, 7000);
    await svc.SaveAsync(SaveDir);                     // state for radius 3
    Assert.False(await svc.TryReuseStateAsync());     // already the one in use

    host.Comet = Spk67P(radius: 2);
    host.Running = true;
    Assert.False(await svc.TryReuseStateAsync());
    host.Running = false;
    Assert.True(await svc.TryReuseStateAsync());
    Assert.Equal(4000ul, host.Timeline.CurrentEpoch!.Nanoseconds);
    Assert.False(await svc.TryReuseStateAsync());     // no repeat for the same configuration

    host.Comet = Spk67P(radius: 5);                   // unknown configuration: nothing to reuse
    Assert.False(await svc.TryReuseStateAsync());
  }

  /// The Earth observer setup is a change: topocentric, and a free look (a tracking look is not,
  /// it re-aims by itself); f64 noise in the look is not.
  [Fact]
  public void Dirty_FollowsTheObserverSetup()
  {
    var host = new FakeSaveFileHost(_root)
    {
      Viewports = [new ViewportDto { Mode = "earth_position", Observer = new ObserverDto { InertialLook = [0, 0, 0, 1] } }],
    };
    var svc = new SaveFileService(host, new Clock());
    Assert.False(svc.IsDirty);
    host.Viewports[0].Observer!.InertialLook = [1e-12, 0, 0, 1];
    Assert.False(svc.IsDirty);
    host.Viewports[0].Observer!.InertialLook = [0.1, 0, 0, 0.995];
    Assert.True(svc.IsDirty);
    host.Viewports[0].Observer!.Orientation = "sun_tracking";
    host.Viewports[0].Observer!.InertialLook = [0, 0, 0, 1];
    Assert.True(svc.IsDirty); // the submode itself
    host.Viewports[0].Observer!.Orientation = "free";
    Assert.False(svc.IsDirty);
    host.Viewports[0].Observer!.Topocentric = true;
    Assert.True(svc.IsDirty);
  }
}
