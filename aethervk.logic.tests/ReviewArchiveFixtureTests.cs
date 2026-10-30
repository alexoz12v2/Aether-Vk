using System;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The reviewer's save as one <c>.avk</c> file written by the Windows build
/// (<c>test_assets/saves/review_67p_2021-12.avk</c>: the four jets, the committed 2021-12-12 →
/// 12-28 range, 1/ρ and trajectories on): the app opens it from its saved state, with nothing
/// recomputed. The engine is the rig's mock (<see cref="SaveArchiveRoundTripTests"/>): it reports
/// the archive's own compatibility JSON and dump version, so the test proves the archive still
/// matches what today's open path accepts, and that the restored bytes are the archived ones.
/// </summary>
public sealed class ReviewArchiveFixtureTests : IDisposable
{
  private delegate bool GetClock(out short centuries, out ulong nanoseconds);

  private readonly string _root = Path.Combine(Path.GetTempPath(), "avk_fixture_" + Guid.NewGuid().ToString("N"));
  private string WorkRoot => Path.Combine(_root, "open");

  public ReviewArchiveFixtureTests() => Directory.CreateDirectory(_root);

  public void Dispose()
  {
    try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
  }

  internal static string FixtureArchive()
  {
    for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
    {
      var p = Path.Combine(d.FullName, "test_assets", "saves", "review_67p_2021-12.avk");
      if (File.Exists(p)) return p;
    }
    throw new FileNotFoundException("test_assets/saves/review_67p_2021-12.avk");
  }

  [Fact]
  public async Task TheWindowsArchive_OpensFromItsState_WithTheFourJetsAndTheCommittedRange()
  {
    // opened from a copy: the fixture itself is never rewritten
    var avk = Path.Combine(_root, "review.avk");
    File.Copy(FixtureArchive(), avk);
    string compatJson, stateFile;
    uint dumpVersion;
    byte[] archivedState;
    using (var zip = ZipFile.OpenRead(avk))
    {
      using var ix = JsonDocument.Parse(zip.GetEntry(SaveIndex.FileName)!.Open());
      var state = ix.RootElement.GetProperty("states").EnumerateArray().Single(s => s.GetProperty("latest").GetBoolean());
      compatJson = state.GetProperty("compatibility_json").GetString()!;
      dumpVersion = state.GetProperty("dump_version").GetUInt32();
      stateFile = state.GetProperty("file").GetString()!;
      using var s = zip.GetEntry(stateFile)!.Open();
      using var m = new MemoryStream();
      s.CopyTo(m);
      archivedState = m.ToArray();
    }
    // today's engine writes the same dump version: an older archive would be recomputed instead
    Assert.Equal(9u, dumpVersion);
    Assert.NotEmpty(archivedState);

    byte[]? restored = null;
    var rig = new AppSaveFileHostTests.Rig();
    rig.Runtime.Setup(r => r.SceneDumpVersion).Returns(dumpVersion);
    rig.Runtime.Setup(r => r.GetCompatibilityJson()).Returns(compatJson);
    rig.Runtime.Setup(r => r.RestoreSceneFromFile(It.IsAny<string>())).Returns<string>(file =>
    {
      restored = File.ReadAllBytes(file);
      rig.Fire(ExternalStateType.SceneRestored, 1, 0);
      return true;
    });
    rig.Runtime.Setup(r => r.SeekEpochSync(It.IsAny<short>(), It.IsAny<ulong>())).Returns(true);
    short c0;
    ulong n0;
    rig.Runtime.Setup(r => r.GetSimulationClock(out c0, out n0))
      .Returns(new GetClock((out short c, out ulong ns) => { c = 0; ns = 693_921_600_000_000_000UL; return true; }));

    var saving = new SaveFileService(rig.Host, new SaveArchiveTests.Clock(), archiveWorkRoot: WorkRoot);
    var r = await saving.OpenAsync(avk);

    Assert.True(r.Success);
    Assert.True(r.StateRestored);
    Assert.False(r.Sought);
    rig.Runtime.Verify(x => x.SeekEpochSync(It.IsAny<short>(), It.IsAny<ulong>()), Times.Never);
    rig.Runtime.Verify(x => x.DumpSceneToFile(It.IsAny<string>()), Times.Never);
    Assert.Equal(archivedState, restored);
    Assert.False(saving.IsDirty);

    // the committed range and the comet
    Assert.True(rig.Comet.IsAlmanacLoaded);
    // compared as instants: the Windows save wrote the end without its "Z"
    static DateTimeOffset Utc(string? s) => DateTimeOffset.Parse(s!, CultureInfo.InvariantCulture,
      DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal);
    Assert.Equal(new DateTimeOffset(2021, 12, 12, 0, 0, 0, TimeSpan.Zero), Utc(rig.Timeline.CommittedStartEpoch));
    Assert.Equal(new DateTimeOffset(2021, 12, 28, 0, 0, 0, TimeSpan.Zero), Utc(rig.Timeline.CommittedEndEpoch));
    var index = saving.CurrentIndex!;
    Assert.Equal((692_539_200_000_000_000UL, 693_921_600_000_000_000UL),
      (index.Timeline.CommittedStart!.Nanoseconds, index.Timeline.CommittedEnd!.Nanoseconds));

    // the four jets (latitude °, speed m/s, Q_d kg/s at 1 AU), all native
    var jets = rig.Model.Jets.Select(j => (j.LatitudeDeg, j.StartVelocityMeanMs, j.DustRate1AuKgs)).ToArray();
    Assert.Equal(
      [(-54f, 70f, 17.178127f), (-67f, 70f, 17.30401f), (-10f, 50f, 12.35723f), (40f, 8f, 1.9738096f)],
      jets);
    Assert.Equal(new uint[] { 12716195, 633654374, 1892684929, 1273352477 }, rig.Model.Jets.Select(j => j.Seed));

    // the dust view options
    Assert.Equal((true, true, 0.0), (rig.Model.DustRhoNorm, rig.Model.DustTrajectories, rig.Model.DustExposureEv));
  }
}
