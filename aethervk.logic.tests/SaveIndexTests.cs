using System;
using System.IO;
using AetherVk.Logic.SaveFile;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>Save index format 1 (docs/savefile.md): JSON shape, round trip, version gate, atomic writes, hash.</summary>
public sealed class SaveIndexTests : IDisposable
{
  private readonly string _dir = Path.Combine(Path.GetTempPath(), "avk_saveindex_" + Guid.NewGuid().ToString("N"));

  public void Dispose()
  {
    if (Directory.Exists(_dir))
      Directory.Delete(_dir, true);
  }

  private static SaveIndex Sample()
  {
    var t = new DateTimeOffset(2026, 10, 12, 21, 30, 0, TimeSpan.Zero);
    return new SaveIndex
    {
      AppVersion = "test",
      CreatedAt = t,
      ModifiedAt = t,
      Timeline = new TimelineDto
      {
        CommittedStart = new EpochParts(0, 812_548_800_000_000_000),
        CommittedEnd = new EpochParts(0, ulong.MaxValue - 1), // u64 precision survives
        CurrentEpoch = new EpochParts(-1, 7),
        StartPaused = true,
      },
      Spk = [new SpkDto { File = "spk/a.bsp", NaifId = 1000012, Committed = true, Sbdb = new SbdbDto { E = 0.64, Tp = null } }],
      States = [new StateDto { CompatHash = "00ff", File = "states/00ff/scene.bin", DumpVersion = 8, Latest = true }],
      Jets = [new JetDto { Seed = 4_000_000_000, StreamColor = [0.1f, 0.2f, 0.3f, 1] }],
      Viewports = [new ViewportDto { Mode = "comet_orbiting", Camera = new CameraSnapshotDto(), OrbitOffsetAu = [0.1, 0, 0] }],
      Billboards = [new BillboardDto { Id = Guid.NewGuid(), TextureKey = "/x.png", RotationDeg = -30 }],
      CometAppearance = new CometAppearanceIndexDto { Mode = "custom", MeshKey = "/m.glb" },
    };
  }

  [Fact]
  public void RoundTrip_PreservesEveryValue_WithSnakeCaseKeysAndEpochArrays()
  {
    var a = Sample();
    SaveIndexIo.Write(_dir, a);
    var json = File.ReadAllText(SaveIndexIo.IndexPath(_dir));
    Assert.Contains("\"format_version\": 1", json);
    Assert.Contains("\"committed_start\": [", json);
    Assert.Contains("\"compat_hash\": \"00ff\"", json);
    Assert.Contains("\"stream_color\"", json);
    var (b, err) = SaveIndexIo.Read(_dir);
    Assert.Equal(SaveIndexError.None, err);
    Assert.Equal(SaveIndexIo.ToJson(a), SaveIndexIo.ToJson(b!));
    Assert.Equal(ulong.MaxValue - 1, b!.Timeline.CommittedEnd!.Nanoseconds);
    Assert.Equal(-1, b.Timeline.CurrentEpoch!.Centuries);
    Assert.Equal(4_000_000_000u, b.Jets[0].Seed);
    Assert.Null(b.Spk[0].Sbdb!.Tp);
  }

  [Fact]
  public void Read_RefusesMissingCorruptAndNewerIndexes()
  {
    Assert.Equal(SaveIndexError.Missing, SaveIndexIo.Read(_dir).Error);
    Directory.CreateDirectory(_dir);
    File.WriteAllText(SaveIndexIo.IndexPath(_dir), "{ not json");
    Assert.Equal(SaveIndexError.Unreadable, SaveIndexIo.Read(_dir).Error);
    File.WriteAllText(SaveIndexIo.IndexPath(_dir), "{\"format_version\": 2}");
    Assert.Equal(SaveIndexError.NewerFormat, SaveIndexIo.Read(_dir).Error);
    File.WriteAllText(SaveIndexIo.IndexPath(_dir), "{\"no_version\": true}");
    Assert.Equal(SaveIndexError.Unreadable, SaveIndexIo.Read(_dir).Error);
  }

  [Fact]
  public void Write_ReplacesAtomically_AndIgnoresAStaleTmp()
  {
    var a = Sample();
    SaveIndexIo.Write(_dir, a);
    // an interrupted previous write left garbage in the .tmp
    File.WriteAllText(SaveIndexIo.IndexPath(_dir) + ".tmp", "garbage");
    Assert.Equal(SaveIndexError.None, SaveIndexIo.Read(_dir).Error);
    a.AppVersion = "second";
    SaveIndexIo.Write(_dir, a);
    Assert.Equal("second", SaveIndexIo.Read(_dir).Index!.AppVersion);
    Assert.False(File.Exists(SaveIndexIo.IndexPath(_dir) + ".tmp"));
  }

  [Theory]
  [InlineData("", "cbf29ce484222325")]
  [InlineData("a", "af63dc4c8601ec8c")]
  [InlineData("foobar", "85944171f73967e8")]
  public void CompatHash_IsFnv1a64(string input, string expected) =>
    Assert.Equal(expected, SaveIndexIo.CompatHash(input));
}
