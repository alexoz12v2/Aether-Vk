using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The reviewer's save through the app pipeline and a single-file save: <see cref="SaveFileService"/>
/// over the real <see cref="AppSaveFileHost"/> and sessions (<see cref="AppSaveFileHostTests.Rig"/>),
/// opened, saved as <c>.avk</c> and opened again in a fresh application.
/// <para>No logic test loads the native cdylib: the rig's runtime is a mock that simulates the
/// engine's callbacks. Not covered here, therefore: the engine's own scene dump / restore (the
/// dumped bytes are the mock's, compared byte for byte through the archive), the compatibility
/// JSON the engine derives from the restored configuration (the mock returns a constant), and the
/// asset imports (the mock refuses them, so the fixture's FITS textures are reported missing and
/// left out of the re-save).</para>
/// </summary>
public sealed class SaveArchiveRoundTripTests : IDisposable
{
  private delegate bool GetClock(out short centuries, out ulong nanoseconds);

  private readonly string _root = Path.Combine(Path.GetTempPath(), "avk_roundtrip_" + Guid.NewGuid().ToString("N"));
  private string WorkRoot => Path.Combine(_root, "open");

  public SaveArchiveRoundTripTests() => Directory.CreateDirectory(_root);

  public void Dispose()
  {
    try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
  }

  private static void CopyDir(string from, string to)
  {
    foreach (var f in Directory.GetFiles(from, "*", SearchOption.AllDirectories))
    {
      var dest = Path.Combine(to, Path.GetRelativePath(from, f));
      Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
      File.Copy(f, dest);
    }
  }

  /// <summary>The engine side of the rig: version-8 dumps of <paramref name="dump"/>, restores
  /// that record what they read, and a clock at <paramref name="clockNs"/>.</summary>
  private static void Engine(AppSaveFileHostTests.Rig rig, byte[] dump, ulong clockNs, Action<string, byte[]>? restored = null)
  {
    rig.Runtime.Setup(r => r.SceneDumpVersion).Returns(8u);
    rig.Runtime.Setup(r => r.DumpSceneToFile(It.IsAny<string>())).Returns<string>(file =>
    {
      File.WriteAllBytes(file, dump);
      rig.Fire(ExternalStateType.SceneDumped, 1, 0);
      return true;
    });
    rig.Runtime.Setup(r => r.RestoreSceneFromFile(It.IsAny<string>())).Returns<string>(file =>
    {
      restored?.Invoke(file, File.ReadAllBytes(file));
      rig.Fire(ExternalStateType.SceneRestored, 1, 0);
      return true;
    });
    rig.Runtime.Setup(r => r.SeekEpochSync(It.IsAny<short>(), It.IsAny<ulong>())).Returns(true);
    short c0;
    ulong n0;
    rig.Runtime.Setup(r => r.GetSimulationClock(out c0, out n0))
      .Returns(new GetClock((out short c, out ulong ns) => { c = 0; ns = clockNs; return true; }));
  }

  [Fact]
  public async Task TheReviewSave_SavedAsAvk_ReopensFromItsState_WithTheSameJetsAndDustView()
  {
    // the fixture is opened from a copy: opening expires stale states and rewrites the index
    var folder = Path.Combine(_root, "review_67p_2021-12");
    CopyDir(ReviewSaveReproTests.FixtureDir(), folder);
    const ulong committedEnd = 693_921_600_000_000_000UL;
    var dump = new byte[1 << 20];
    new Random(67).NextBytes(dump);

    // 1. the app opens the folder save (its latest state file is not in the fixture: a seek)
    var first = new AppSaveFileHostTests.Rig();
    Engine(first, dump, committedEnd);
    var saving = new SaveFileService(first.Host, new SaveArchiveTests.Clock(), archiveWorkRoot: WorkRoot);
    var opened = await saving.OpenAsync(folder);
    Assert.True(opened.Success);
    Assert.True(first.Comet.IsAlmanacLoaded);
    Assert.Equal(4, first.Model.Jets.Count);
    var rates = first.Model.Jets.Select(j => j.DustRate1AuKgs).ToArray();
    Assert.All(rates, q => Assert.True(q > 0, $"Q_d {q}"));

    // non-default dust view options
    var defaults = new ModelSession();
    Assert.False(defaults.DustRhoNorm || defaults.DustTrajectories);
    Assert.Equal(0.0, defaults.DustExposureEv);
    first.Model.DustRhoNorm = true;
    first.Model.DustTrajectories = true;
    first.Model.DustExposureEv = 1.75;

    // 2. Save As one file
    var avk = Path.Combine(_root, "review.avk");
    Assert.True(await saving.SaveAsync(avk));
    Assert.Equal(Path.GetFullPath(avk), saving.CurrentSavePath);
    var savedState = Assert.Single(Directory.GetFiles(Path.Combine(saving.CurrentSaveDir!, "states"), "scene.bin",
      SearchOption.AllDirectories));
    Assert.Equal(dump, File.ReadAllBytes(savedState));

    // 3. a fresh application opens it: the state is restored, nothing is recomputed
    string? restoredFile = null;
    byte[]? restoredBytes = null;
    var second = new AppSaveFileHostTests.Rig();
    Engine(second, [], 0, (f, b) => (restoredFile, restoredBytes) = (f, b));
    var reopening = new SaveFileService(second.Host, new SaveArchiveTests.Clock(), archiveWorkRoot: WorkRoot);
    var r = await reopening.OpenAsync(avk);

    Assert.True(r.Success);
    Assert.True(r.StateRestored);
    Assert.False(r.Sought);
    second.Runtime.Verify(x => x.SeekEpochSync(It.IsAny<short>(), It.IsAny<ulong>()), Times.Never);
    second.Runtime.Verify(x => x.DumpSceneToFile(It.IsAny<string>()), Times.Never);
    Assert.NotEqual(saving.CurrentSaveDir, reopening.CurrentSaveDir);
    Assert.StartsWith(WorkRoot, reopening.CurrentSaveDir);
    Assert.False(reopening.IsDirty);

    // the restored file is the extracted one, byte for byte the dump
    var extractedState = Path.Combine(reopening.CurrentSaveDir!, Path.GetRelativePath(saving.CurrentSaveDir!, savedState));
    Assert.Equal(Path.GetFullPath(extractedState), Path.GetFullPath(restoredFile!));
    Assert.Equal(dump, restoredBytes);
    Assert.Equal(dump, File.ReadAllBytes(extractedState));
    Assert.Equal(SaveArchiveTests.Tree(saving.CurrentSaveDir!), SaveArchiveTests.Tree(reopening.CurrentSaveDir!));

    // the jets and the dust view
    Assert.True(second.Comet.IsAlmanacLoaded);
    Assert.Equal(rates, second.Model.Jets.Select(j => j.DustRate1AuKgs).ToArray());
    Assert.Equal(first.Model.Jets.Select(j => j.Seed), second.Model.Jets.Select(j => j.Seed));
    Assert.Equal((true, true, 1.75), (second.Model.DustRhoNorm, second.Model.DustTrajectories, second.Model.DustExposureEv));
    var index = reopening.CurrentIndex!;
    Assert.Equal((true, true, 1.75), (index.Model.DustRhoNorm, index.Model.DustTrajectories, index.Model.DustExposureEv));
    Assert.Equal(committedEnd, index.Timeline.CurrentEpoch!.Nanoseconds);
  }
}
